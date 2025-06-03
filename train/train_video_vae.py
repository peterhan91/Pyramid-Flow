import sys
import os
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset import (
    ImageDataset,
    VideoDataset,
    create_mixed_dataloaders,
)

try:
    from dataset.text_video_dataset import TextVideoDataset
    TEXT_VIDEO_DATASET_AVAILABLE = True
except ImportError:
    TEXT_VIDEO_DATASET_AVAILABLE = False
    print("TextVideoDataset not available - using original VideoDataset")

from trainer_misc import (
    NativeScalerWithGradNormCount,
    create_optimizer,
    train_one_epoch,
    auto_load_model,
    save_model,
    init_distributed_mode,
    cosine_scheduler,
)

from video_vae import CausalVideoVAELossWrapper, WanVideoVAELossWrapper
import utils


def evaluate_model(model, data_loader, device, args, epoch, model_dtype='bf16'):
    """
    Evaluate the model on validation set and compute reconstruction metrics.
    """
    model.eval()
    
    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16
    
    total_vae_loss = 0.0
    total_gan_loss = 0.0
    total_samples = 0
    detailed_losses = {}
    
    with torch.no_grad():
        eval_count = 0
        for samples in data_loader:
            if eval_count >= args.max_eval_videos:
                break
                
            samples['video'] = samples['video'].to(device, non_blocking=True)
            batch_size = samples['video'].shape[0]
            
            with torch.cuda.amp.autocast(enabled=True, dtype=_dtype):
                rec_loss, gan_loss, log_losses = model(samples['video'], args.global_step, identifier=samples['identifier'])
            
            if rec_loss is not None:
                total_vae_loss += rec_loss.item() * batch_size
            if gan_loss is not None:
                total_gan_loss += gan_loss.item() * batch_size
                
            # Accumulate detailed losses
            for key, value in log_losses.items():
                if key not in detailed_losses:
                    detailed_losses[key] = 0.0
                detailed_losses[key] += value * batch_size
                
            total_samples += batch_size
            eval_count += batch_size
    
    # Compute averages
    avg_vae_loss = total_vae_loss / total_samples if total_samples > 0 else 0.0
    avg_gan_loss = total_gan_loss / total_samples if total_samples > 0 else 0.0
    
    avg_detailed_losses = {}
    for key, value in detailed_losses.items():
        avg_detailed_losses[f'eval_{key}'] = value / total_samples
    
    eval_stats = {
        'eval_vae_loss': avg_vae_loss,
        'eval_gan_loss': avg_gan_loss,
        **avg_detailed_losses
    }
    
    model.train()
    return eval_stats


def save_reconstruction_samples(model, data_loader, device, args, epoch, save_dir, model_dtype='bf16'):
    """
    Save reconstruction samples for visual inspection with side-by-side comparisons.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    model.eval()
    
    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16
    
    save_path = os.path.join(save_dir, f'reconstructions_epoch_{epoch}')
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        sample_count = 0
        for samples in data_loader:
            if sample_count >= args.num_eval_samples:
                break
                
            videos = samples['video'].to(device, non_blocking=True)
            
            # Take only the first video from the batch for reconstruction
            video = videos[:1]  # Shape: [1, C, T, H, W]
            
            try:
                # Use the wrapper's reconstruct method
                with torch.cuda.amp.autocast(enabled=True, dtype=_dtype):
                    if hasattr(model, 'module'):
                        model_unwrapped = model.module
                    else:
                        model_unwrapped = model
                        
                    # Reconstruct with both deterministic and stochastic sampling
                    pil_images_det = model_unwrapped.reconstruct(video, sample=False)
                    pil_images_sample = model_unwrapped.reconstruct(video, sample=True)
                
                # Convert original frames to PIL images
                original_frames = video[0].cpu()  # [C, T, H, W]
                original_frames = (original_frames / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
                
                original_pil_images = []
                for t in range(original_frames.shape[1]):
                    frame = original_frames[:, t, :, :]  # [C, H, W]
                    frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
                    frame_np = (frame_np * 255).astype(np.uint8)
                    pil_img = Image.fromarray(frame_np)
                    original_pil_images.append(pil_img)
                
                # Create side-by-side comparisons
                num_frames = min(len(original_pil_images), len(pil_images_det), len(pil_images_sample), 8)
                
                for t in range(num_frames):
                    orig_img = original_pil_images[t]
                    det_img = pil_images_det[t]
                    sample_img = pil_images_sample[t]
                    
                    # Get dimensions
                    width, height = orig_img.size
                    
                    # Create composite image: Original | Deterministic | Stochastic
                    # Add some padding and labels
                    label_height = 30
                    padding = 10
                    composite_width = width * 3 + padding * 4
                    composite_height = height + label_height + padding * 2
                    
                    composite = Image.new('RGB', (composite_width, composite_height), color='white')
                    
                    # Paste images
                    composite.paste(orig_img, (padding, label_height + padding))
                    composite.paste(det_img, (width + padding * 2, label_height + padding))
                    composite.paste(sample_img, (width * 2 + padding * 3, label_height + padding))
                    
                    # Add labels
                    try:
                        # Try to use a default font, fall back to default if not available
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except:
                        try:
                            font = ImageFont.truetype("arial.ttf", 16)
                        except:
                            font = ImageFont.load_default()
                    
                    draw = ImageDraw.Draw(composite)
                    
                    # Label positions
                    label_y = 5
                    draw.text((padding + width//2 - 30, label_y), "Original", fill='black', font=font)
                    draw.text((width + padding * 2 + width//2 - 50, label_y), "Deterministic", fill='black', font=font)
                    draw.text((width * 2 + padding * 3 + width//2 - 40, label_y), "Stochastic", fill='black', font=font)
                    
                    # Save composite image
                    composite_path = os.path.join(save_path, f'sample_{sample_count}_comparison_frame_{t}.png')
                    composite.save(composite_path)
                
                # Also create a video strip showing all frames side by side
                if num_frames > 1:
                    create_video_strip(original_pil_images[:num_frames], 
                                     pil_images_det[:num_frames], 
                                     pil_images_sample[:num_frames],
                                     save_path, sample_count)
                
                print(f"Saved reconstruction comparison {sample_count} at epoch {epoch}")
                
            except Exception as e:
                print(f"Failed to save reconstruction sample {sample_count}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
            sample_count += 1
    
    model.train()
    print(f"Saved {sample_count} reconstruction comparison samples to {save_path}")


def create_video_strip(original_frames, det_frames, sample_frames, save_path, sample_idx):
    """
    Create a strip showing all frames of a video sequence side by side.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    if not original_frames:
        return
    
    # Get dimensions
    width, height = original_frames[0].size
    num_frames = len(original_frames)
    
    # Create strip: 3 rows (original, det, sample) x num_frames columns
    label_height = 25
    padding = 5
    strip_width = width * num_frames + padding * (num_frames + 1)
    strip_height = height * 3 + label_height + padding * 4
    
    strip = Image.new('RGB', (strip_width, strip_height), color='white')
    
    # Add row labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(strip)
    
    # Row labels
    row_labels = ["Original", "Deterministic", "Stochastic"]
    for i, label in enumerate(row_labels):
        y_pos = label_height + padding + i * (height + padding) + height // 2 - 10
        draw.text((5, y_pos), label, fill='black', font=font)
    
    # Paste frames
    for frame_idx in range(num_frames):
        x_offset = padding * (frame_idx + 1) + width * frame_idx
        
        # Original frame
        y_offset = label_height + padding
        strip.paste(original_frames[frame_idx], (x_offset, y_offset))
        
        # Deterministic reconstruction
        y_offset = label_height + padding * 2 + height
        strip.paste(det_frames[frame_idx], (x_offset, y_offset))
        
        # Stochastic reconstruction  
        y_offset = label_height + padding * 3 + height * 2
        strip.paste(sample_frames[frame_idx], (x_offset, y_offset))
        
        # Add frame number at top
        draw.text((x_offset + width//2 - 10, 5), f"F{frame_idx}", fill='black', font=font)
    
    # Save video strip
    strip_path = os.path.join(save_path, f'sample_{sample_idx}_video_strip.png')
    strip.save(strip_path)


def log_detailed_metrics(log_stats, epoch, args):
    """
    Enhanced logging with detailed metrics tracking.
    """
    if not utils.is_main_process():
        return
        
    print(f"\n=== Epoch {epoch + 1} Summary ===")
    
    # Training metrics
    train_metrics = {k: v for k, v in log_stats.items() if k.startswith('train_')}
    if train_metrics:
        print("Training Metrics:")
        for key, value in train_metrics.items():
            metric_name = key.replace('train_', '')
            print(f"  {metric_name}: {value:.6f}")
    
    # Evaluation metrics
    eval_metrics = {k: v for k, v in log_stats.items() if k.startswith('eval_')}
    if eval_metrics:
        print("Evaluation Metrics:")
        for key, value in eval_metrics.items():
            metric_name = key.replace('eval_', '')
            print(f"  {metric_name}: {value:.6f}")
    
    # Model info
    if 'n_parameters' in log_stats:
        print(f"Model Parameters: {log_stats['n_parameters'] / 1e6:.2f}M")
    
    print("=" * 40)
    
    # Save detailed metrics to file
    if args.output_dir:
        metrics_file = os.path.join(args.output_dir, "detailed_metrics.json")
        
        # Load existing metrics or create new
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Add current epoch metrics
        current_metrics = {
            'epoch': epoch + 1,
            'global_step': args.global_step,
            **log_stats
        }
        all_metrics.append(current_metrics)
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)


def compute_reconstruction_metrics(model, data_loader, device, args, model_dtype='bf16'):
    """
    Compute additional reconstruction quality metrics (PSNR, SSIM, etc.).
    """
    try:
        import torch.nn.functional as F
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
    except ImportError:
        print("Warning: scikit-image not available, skipping advanced metrics")
        return {}
    
    model.eval()
    
    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        sample_count = 0
        for samples in data_loader:
            if sample_count >= 10:  # Compute metrics on limited samples for efficiency
                break
                
            videos = samples['video'].to(device, non_blocking=True)
            video = videos[:1]  # Take first video
            
            try:
                with torch.cuda.amp.autocast(enabled=True, dtype=_dtype):
                    if hasattr(model, 'module'):
                        model_unwrapped = model.module
                    else:
                        model_unwrapped = model
                        
                    # Get reconstruction
                    latent = model_unwrapped.encode(video, sample=False)
                    recon = model_unwrapped.decode(latent)
                
                # Compute metrics per frame
                original = (video[0].cpu().float() / 2 + 0.5).clamp(0, 1)  # [C, T, H, W]
                reconstructed = (recon[0].cpu().float() / 2 + 0.5).clamp(0, 1)  # [C, T, H, W]
                
                for t in range(min(original.shape[1], reconstructed.shape[1])):
                    orig_frame = original[:, t, :, :].permute(1, 2, 0).numpy()  # [H, W, C]
                    recon_frame = reconstructed[:, t, :, :].permute(1, 2, 0).numpy()  # [H, W, C]
                    
                    # PSNR
                    mse = np.mean((orig_frame - recon_frame) ** 2)
                    if mse > 0:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                        psnr_values.append(psnr)
                    
                    # SSIM (convert to grayscale for efficiency)
                    orig_gray = np.mean(orig_frame, axis=2)
                    recon_gray = np.mean(recon_frame, axis=2)
                    ssim_val = ssim(orig_gray, recon_gray, data_range=1.0)
                    ssim_values.append(ssim_val)
                
            except Exception as e:
                print(f"Failed to compute metrics for sample {sample_count}: {str(e)}")
                continue
                
            sample_count += 1
    
    model.train()
    
    metrics = {}
    if psnr_values:
        metrics['reconstruction_psnr'] = np.mean(psnr_values)
    if ssim_values:
        metrics['reconstruction_ssim'] = np.mean(ssim_values)
    
    return metrics


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process Training script for Video VAE', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Evaluation and logging parameters
    parser.add_argument('--eval_freq', default=5, type=int, help='Evaluate every N epochs')
    parser.add_argument('--save_reconstruction_freq', default=10, type=int, help='Save reconstruction samples every N epochs')
    parser.add_argument('--num_eval_samples', default=4, type=int, help='Number of samples to use for evaluation')
    parser.add_argument('--max_eval_videos', default=50, type=int, help='Maximum number of videos to evaluate')

    # Model parameters
    parser.add_argument('--ema_update', action='store_true')
    parser.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')

    parser.add_argument('--model_path', default='', type=str, help='The vae weight path')
    parser.add_argument('--model_dtype', default='bf16', help="The Model Dtype: bf16 or df16")

    # Add model architecture selection
    parser.add_argument('--model_arch', default='causal_vae', choices=['causal_vae', 'wan_vae'], 
                        type=str, help='Specify VAE architecture: causal_vae or wan_vae')

    # Add WanVAE specific model parameters (optional, with defaults from VideoVAE_)
    parser.add_argument('--wan_vae_dim', default=96, type=int, help='Dimension for WanVAE')
    parser.add_argument('--wan_vae_z_dim', default=16, type=int, help='Latent dimension for WanVAE')
    parser.add_argument('--wan_vae_dim_mult', default=[1, 2, 4, 4], type=int, nargs='+', 
                        help='Dimension multipliers for WanVAE blocks')
    parser.add_argument('--wan_vae_num_res_blocks', default=2, type=int, 
                        help='Number of residual blocks in WanVAE')

    # Using the context parallel to distribute multiple video clips to different devices
    parser.add_argument('--use_context_parallel', action='store_true')
    parser.add_argument('--context_size', default=2, type=int, help="The context length size")
    parser.add_argument('--resolution', default=256, type=int, help="The input resolution for VAE training")
    parser.add_argument('--max_frames', default=24, type=int, help='number of max video frames')
    parser.add_argument('--use_image_video_mixed_training', action='store_true', help="Whether to use the mixed image and video training")

    # The loss weights
    parser.add_argument('--lpips_ckpt', default="/home/jinyang06/models/vae/video_vae_baseline/vgg_lpips.pth", type=str, help="The LPIPS checkpoint path")
    parser.add_argument('--disc_start', default=0, type=int, help="The start iteration for adding GAN Loss")
    parser.add_argument('--logvar_init', default=0.0, type=float, help="The log var init" )
    parser.add_argument('--kl_weight', default=1e-6, type=float, help="The KL loss weight")
    parser.add_argument('--pixelloss_weight', default=1.0, type=float, help="The pixel reconstruction loss weight")
    parser.add_argument('--perceptual_weight', default=1.0, type=float, help="The perception loss weight")
    parser.add_argument('--disc_weight', default=0.1, type=float,  help="The GAN loss weight")
    parser.add_argument('--pretrained_vae_weight', default='', type=str, help='The pretrained vae ckpt path')  
    parser.add_argument('--not_add_normalize', action='store_true')
    parser.add_argument('--add_discriminator', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--lr_disc', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5) of the discriminator')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--image_anno', default='', type=str, help="The image data annotation file path")
    parser.add_argument('--video_anno', default='', type=str, help="The video data annotation file path")
    parser.add_argument('--image_mix_ratio', default=0.1, type=float, help="The image data proportion in the training batch")

    # Add support for TextVideoDataset
    parser.add_argument('--use_text_video_dataset', action='store_true', 
                        help='Use TextVideoDataset instead of original VideoDataset')
    parser.add_argument('--text_video_base_path', default='', type=str, 
                        help='Base path for TextVideoDataset video files')
    parser.add_argument('--text_video_metadata', default='', type=str, 
                        help='Metadata CSV file path for TextVideoDataset')
    parser.add_argument('--frame_interval', default=1, type=int, 
                        help='Frame interval for TextVideoDataset sampling')

    # Distributed Training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true', default=False)
    
    parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--global_step', default=0, type=int, metavar='N', help='The global optimization step')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def build_model(args):
    model_dtype = args.model_dtype
    model_path = args.model_path

    print(f"Using VAE architecture: {args.model_arch}")

    if args.model_arch == 'causal_vae':
        print(f"Load the base CausalVideoVAE checkpoint from path: {model_path}, using dtype {model_dtype}")
        model = CausalVideoVAELossWrapper(
            model_path,
            model_dtype='fp32',      # For training, we used mixed training
            disc_start=args.disc_start,
            logvar_init=args.logvar_init,
            kl_weight=args.kl_weight,
            pixelloss_weight=args.pixelloss_weight,
            perceptual_weight=args.perceptual_weight,
            disc_weight=args.disc_weight,
            interpolate=False,
            add_discriminator=args.add_discriminator,
            freeze_encoder=args.freeze_encoder,
            load_loss_module=True,
            lpips_ckpt=args.lpips_ckpt,
        )
        if args.pretrained_vae_weight: # This was specific to CausalVideoVAE loading its own VAE part
            pretrained_vae_weight = args.pretrained_vae_weight
            print(f"Loading the Causal VAE vae part checkpoint from {pretrained_vae_weight}")
            pass


    elif args.model_arch == 'wan_vae':
        print(f"Initializing WanVideoVAE (VideoVAE_). Model path for weights: {args.model_path}")
        model = WanVideoVAELossWrapper(
            vae_dim=args.wan_vae_dim,
            vae_z_dim=args.wan_vae_z_dim,
            vae_dim_mult=args.wan_vae_dim_mult,
            vae_num_res_blocks=args.wan_vae_num_res_blocks,
            model_dtype=model_dtype, # Pass dtype, though VideoVAE_ doesn't use it in init
            disc_start=args.disc_start,
            logvar_init=args.logvar_init,
            kl_weight=args.kl_weight,
            pixelloss_weight=args.pixelloss_weight,
            perceptual_weight=args.perceptual_weight,
            disc_weight=args.disc_weight,
            add_discriminator=args.add_discriminator,
            freeze_encoder=args.freeze_encoder, # Passed via kwargs
            # load_loss_module is implicit as LPIPSWithDiscriminator is always created
            lpips_ckpt=args.lpips_ckpt,
        )

        if args.model_path and not args.resume: # Only if not resuming a full training state
             print(f"Loading VAE (VideoVAE_) weights from --model_path: {args.model_path} for WanVAE.")
             model.load_checkpoint(args.model_path, strict=False) # Use strict=False if it's just VAE weights

    else:
        raise ValueError(f"Unknown model_arch: {args.model_arch}")


    return model


def main(args):
    init_distributed_mode(args)

    # If enabled, distribute multiple video clips to different devices
    if args.use_context_parallel:
        utils.initialize_context_parallel(args.context_size)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    model = build_model(args)
    
    world_size = utils.get_world_size()
    global_rank = utils.get_rank()

    num_training_steps_per_epoch = args.iters_per_epoch
    log_writer = None

    # building dataset and dataloaders
    image_gpus = max(1, int(world_size * args.image_mix_ratio))
    if args.use_image_video_mixed_training:
        video_gpus = world_size - image_gpus
    else:
        # only use video data
        video_gpus = world_size
        image_gpus = 0

    if global_rank < video_gpus:
        if args.use_text_video_dataset and TEXT_VIDEO_DATASET_AVAILABLE:
            # Use TextVideoDataset
            if not args.text_video_base_path or not args.text_video_metadata:
                raise ValueError("--text_video_base_path and --text_video_metadata are required when using --use_text_video_dataset")
            
            print(f"Using TextVideoDataset with base_path: {args.text_video_base_path}, metadata: {args.text_video_metadata}")
            training_dataset = TextVideoDataset(
                base_path=args.text_video_base_path,
                metadata_path=args.text_video_metadata,
                resolution=args.resolution,
                max_frames=args.max_frames,
                frame_interval=args.frame_interval,
                add_normalize=not args.not_add_normalize
            )
        else:
            # Use original VideoDataset
            training_dataset = VideoDataset(args.video_anno, resolution=args.resolution, 
                max_frames=args.max_frames, add_normalize=not args.not_add_normalize)
    else:
        training_dataset = ImageDataset(args.image_anno, resolution=args.resolution, 
            max_frames=args.max_frames // 4, add_normalize=not args.not_add_normalize)

    data_loader_train = create_mixed_dataloaders(
        training_dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        epoch=args.seed,
        world_size=world_size,
        rank=global_rank,
        image_mix_ratio=args.image_mix_ratio,
        use_image_video_mixed_training=args.use_image_video_mixed_training,
    )
    
    # Create validation dataloader (reuse training dataset with smaller batch size for eval)
    data_loader_val = None
    if not args.disable_eval:
        # Use a subset of training data for validation (in real setup, use separate validation set)
        if global_rank < video_gpus:
            if args.use_text_video_dataset and TEXT_VIDEO_DATASET_AVAILABLE:
                # Use TextVideoDataset for validation
                val_dataset = TextVideoDataset(
                    base_path=args.text_video_base_path,
                    metadata_path=args.text_video_metadata,
                    resolution=args.resolution,
                    max_frames=args.max_frames,
                    frame_interval=args.frame_interval,
                    add_normalize=not args.not_add_normalize
                )
            else:
                # Use original VideoDataset for validation
                val_dataset = VideoDataset(args.video_anno, resolution=args.resolution, 
                    max_frames=args.max_frames, add_normalize=not args.not_add_normalize)
        else:
            val_dataset = ImageDataset(args.image_anno, resolution=args.resolution, 
                max_frames=args.max_frames // 4, add_normalize=not args.not_add_normalize)
                
        data_loader_val = create_mixed_dataloaders(
            val_dataset,
            batch_size=min(4, args.batch_size),  # Smaller batch size for evaluation
            num_workers=args.num_workers,
            epoch=args.seed + 1000,  # Different seed for validation
            world_size=world_size,
            rank=global_rank,
            image_mix_ratio=args.image_mix_ratio,
            use_image_video_mixed_training=args.use_image_video_mixed_training,
        )
    
    torch.distributed.barrier()

    model.to(device)
    model_without_ddp = model

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(name)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp.vae)
    optimizer_disc = create_optimizer(args, model_without_ddp.loss.discriminator) if args.add_discriminator else None

    loss_scaler = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False)
    loss_scaler_disc = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False) if args.add_discriminator else None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    lr_schedule_values_disc = cosine_scheduler(
        args.lr_disc, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    ) if args.add_discriminator else None

    auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, 
        loss_scaler=loss_scaler, optimizer_disc=optimizer_disc,
    )
    
    print(f"Start training for {args.epochs} epochs, the global iterations is {args.global_step}")
    start_time = time.time()
    torch.distributed.barrier()
            
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(
            model, 
            args.model_dtype,
            data_loader_train,
            optimizer, 
            optimizer_disc,
            device, 
            epoch, 
            loss_scaler,
            loss_scaler_disc,
            args.clip_grad, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            lr_schedule_values_disc=lr_schedule_values_disc,
            args=args,
            print_freq=args.print_freq,
            iters_per_epoch=num_training_steps_per_epoch,
        )

        # Evaluation
        eval_stats = {}
        if not args.disable_eval and data_loader_val is not None and (epoch + 1) % args.eval_freq == 0:
            print(f"Running evaluation at epoch {epoch + 1}...")
            eval_stats = evaluate_model(model, data_loader_val, device, args, epoch, args.model_dtype)
            
            # Compute additional reconstruction quality metrics
            if utils.is_main_process():
                print("Computing reconstruction quality metrics...")
                quality_metrics = compute_reconstruction_metrics(model, data_loader_val, device, args, args.model_dtype)
                eval_stats.update(quality_metrics)
            
            if utils.is_main_process():
                print(f"Evaluation results at epoch {epoch + 1}:")
                for key, value in eval_stats.items():
                    print(f"  {key}: {value:.6f}")
        
        # Save reconstruction samples
        if not args.disable_eval and data_loader_val is not None and args.output_dir and \
           (epoch + 1) % args.save_reconstruction_freq == 0 and utils.is_main_process():
            print(f"Saving reconstruction samples at epoch {epoch + 1}...")
            save_reconstruction_samples(
                model, data_loader_val, device, args, epoch + 1, 
                args.output_dir, args.model_dtype
            )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq, optimizer_disc=optimizer_disc
                )
        
        # Combine train and eval stats for logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **eval_stats,  # eval stats already have 'eval_' prefix
            'epoch': epoch, 
            'n_parameters': n_learnable_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        log_detailed_metrics(log_stats, epoch, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
