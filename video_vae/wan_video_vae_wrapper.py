import torch
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
from einops import rearrange

from .modelling_wanvae import VideoVAE_ # Main VAE model
from .modeling_loss import LPIPSWithDiscriminator
from video_vae.modeling_enc_dec import DiagonalGaussianDistribution # Re-using this for posterior

class WanVideoVAELossWrapper(nn.Module):
    """
    Wrapper for WanVideoVAE (specifically VideoVAE_) to integrate with the training script,
    handling loss computation.
    """
    def __init__(self,
                 # Model specific hyperparams for VideoVAE_
                 vae_dim: int = 96,
                 vae_z_dim: int = 16,
                 vae_dim_mult: list = [1, 2, 4, 4],
                 vae_num_res_blocks: int = 2,
                 # Loss specific params
                 disc_start=0, 
                 logvar_init=0.0, 
                 kl_weight=1.0,
                 pixelloss_weight=1.0, 
                 perceptual_weight=1.0, 
                 disc_weight=0.5,
                 add_discriminator=True, 
                 lpips_ckpt=None,
                 model_dtype='fp32', # Added for consistency, though VideoVAE_ doesn't explicitly use it in init
                 **kwargs, # To catch any other args from train_script like freeze_encoder
                ):
        super().__init__()

        self.vae = VideoVAE_(
            dim=vae_dim,
            z_dim=vae_z_dim,
            dim_mult=vae_dim_mult,
            num_res_blocks=vae_num_res_blocks,
            # attn_scales and dropout are default in VideoVAE_
        )
        
        # Store the scale factors, copying from WanVideoVAE class in modelling_wanvae.py
        # These are crucial for the VideoVAE_ encode/decode methods.
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # Ensure these are tensors. They will be moved to device by VideoVAE_ methods.
        self.scale_mean = torch.tensor(mean)
        self.scale_std = torch.tensor(std)
        # The scale format expected by VideoVAE_.encode/decode is [mean_tensor, inv_std_tensor]
        self.training_scale = [self.scale_mean, 1.0 / self.scale_std]

        # Placeholder for VAE scale factor if needed by LPIPS or other parts, like in CausalWrapper
        self.vae_scale_factor = 0.18215 # Default from AutoencoderKL, adjust if WanVAE has a specific one

        # Handle freeze_encoder if passed
        freeze_encoder = kwargs.get('freeze_encoder', False)
        if freeze_encoder:
            print("Freeze the parameters of WanVideoVAE encoder")
            # VideoVAE_ has self.encoder
            for parameter in self.vae.encoder.parameters():
                parameter.requires_grad = False
            # VideoVAE_ has self.conv1 after encoder, which produces mu/logvar
            # If this is part of "quant_conv" equivalent
            for parameter in self.vae.conv1.parameters():
                 parameter.requires_grad = False


        self.add_discriminator = add_discriminator

        # Loss module (reusing LPIPSWithDiscriminator)
        self.loss = LPIPSWithDiscriminator(
            disc_start, 
            logvar_init=logvar_init, 
            kl_weight=kl_weight,
            pixelloss_weight=pixelloss_weight, 
            perceptual_weight=perceptual_weight, 
            disc_weight=disc_weight,
            add_discriminator=add_discriminator, 
            using_3d_discriminator=False, # Or True, depending on what's appropriate
            disc_num_layers=4, 
            lpips_ckpt=lpips_ckpt
        )
        
        self.disc_start = disc_start

    def load_checkpoint(self, checkpoint_path, strict=False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle nested checkpoints (e.g. from a full training state)
        if 'model' in checkpoint: # Common pattern for top-level model state
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint: # Another common pattern
            checkpoint = checkpoint['state_dict']

        vae_checkpoint = OrderedDict()
        disc_checkpoint = OrderedDict()
        
        # Check if weights are already prefixed with 'vae.' or 'loss.discriminator.'
        has_vae_prefix = any(key.startswith('vae.') for key in checkpoint.keys())
        has_loss_prefix = any(key.startswith('loss.discriminator.') for key in checkpoint.keys())

        # If no prefixes, assume all weights are for VAE, or need more specific loading
        # This part might need adjustment based on how WanVideoVAE checkpoints are saved.
        # The VideoVAE_ in modelling_wanvae.py does not have submodules named 'vae' or 'loss' directly.
        # Its own modules are self.encoder, self.conv1, self.decoder.
        # The LPIPSWithDiscriminator has self.discriminator.
        
        for key, value in checkpoint.items():
            if key.startswith('vae.'): # If main script saved it with 'vae.' prefix
                new_key = key.split('vae.', 1)[1]
                vae_checkpoint[new_key] = value
            elif key.startswith('model.'): # If WanVideoVAE's own converter was used, it adds 'model.'
                 new_key = key.split('model.', 1)[1]
                 vae_checkpoint[new_key] = value
            elif key.startswith('loss.discriminator.'): # If main script saved it with this prefix
                new_key = key.split('loss.discriminator.', 1)[1]
                disc_checkpoint[new_key] = value
            elif not has_vae_prefix and not has_loss_prefix: 
                # Fallback: assume it's for the VAE (VideoVAE_ model) directly if no prefix matches
                # This might be risky if checkpoint contains mixed weights without clear prefixes.
                # For VideoVAE_, direct child modules are encoder, conv1, conv2, decoder
                is_vae_submodule = any(key.startswith(sub) for sub in ['encoder.', 'conv1.', 'conv2.', 'decoder.'])
                if is_vae_submodule:
                     vae_checkpoint[key] = value
                # Add more specific rules if discriminator weights are also unprefixed
                # For now, assume discriminator weights will have a 'loss.discriminator.' prefix or similar

        if vae_checkpoint:
            load_result = self.vae.load_state_dict(vae_checkpoint, strict=strict)
            print(f"Loaded VAE (VideoVAE_) weights from {checkpoint_path}. Load result: {load_result}")
        else:
            print(f"No VAE weights found or loaded from {checkpoint_path} for VideoVAE_.")

        if self.add_discriminator and disc_checkpoint:
            load_result_disc = self.loss.discriminator.load_state_dict(disc_checkpoint, strict=strict)
            print(f"Loaded Discriminator weights from {checkpoint_path}. Load result: {load_result_disc}")
        elif self.add_discriminator:
            print(f"No Discriminator weights found or loaded from {checkpoint_path}.")


    def forward(self, x, step, identifier=['video']):
        # Use VideoVAE_'s built-in methods with proper scale handling
        # First encode to get latent
        z = self.vae.encode(x, scale=self.training_scale)
        
        # Get mu and log_var by replicating the encode process
        self.vae.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        # Encode with chunking to get both mu and log_var
        for i in range(iter_):
            self.vae._enc_conv_idx = [0]
            if i == 0:
                out = self.vae.encoder(x[:, :, :1, :, :],
                                   feat_cache=self.vae._enc_feat_map,
                                   feat_idx=self.vae._enc_conv_idx)
            else:
                out_ = self.vae.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                    feat_cache=self.vae._enc_feat_map,
                                    feat_idx=self.vae._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        
        # Get mu and log_var from conv1 (before scale normalization)
        conv1_out = self.vae.conv1(out)
        mu_raw, log_var = conv1_out.chunk(2, dim=1)
        
        # Apply scale normalization to mu_raw to get the actual mu (matches what encode() returns)
        scale_mean = self.scale_mean.to(dtype=mu_raw.dtype, device=mu_raw.device)
        scale_inv_std = (1.0 / self.scale_std).to(dtype=mu_raw.dtype, device=mu_raw.device)
        mu = (mu_raw - scale_mean.view(1, self.vae.z_dim, 1, 1, 1)) * scale_inv_std.view(1, self.vae.z_dim, 1, 1, 1)
        
        # Decode using the encoded latent
        x_recon = self.vae.decode(z, scale=self.training_scale)

        # Create DiagonalGaussianDistribution for the LPIPSWithDiscriminator
        # mu, log_var from VideoVAE_ are [B, Z_DIM, T', H', W']
        # DiagonalGaussianDistribution expects [B, 2*Z_DIM, T', H', W']
        parameters = torch.cat((mu, log_var), dim=1)
        posterior = DiagonalGaussianDistribution(parameters)

        # The reconstruct loss (and GAN G loss)
        reconstruct_loss, rec_log = self.loss(
            x, x_recon, posterior,
            optimizer_idx=0, global_step=step, last_layer=self.get_last_layer(),
        )

        # The loss to train the discriminator (GAN D loss)
        gan_loss = None
        gan_log = {}
        if self.add_discriminator and step >= self.disc_start:
            gan_loss, gan_log = self.loss(
                x, x_recon, posterior, optimizer_idx=1,
                global_step=step, last_layer=self.get_last_layer(),
            )

        loss_log = {**rec_log, **gan_log}
        return reconstruct_loss, gan_loss, loss_log

    # Required by LPIPSWithDiscriminator for perceptual loss
    def get_last_layer(self):
        # This should return the weights of the last convolutional layer of the decoder.
        # In VideoVAE_, the decoder is self.vae.decoder, and its last conv is self.vae.decoder.head[-1]
        # which is a CausalConv3d instance from modelling_wanvae.py
        return self.vae.decoder.head[-1].weight
        
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode(self, x, sample=False, is_init_image=True, 
            temporal_chunk=False, window_size=16, tile_sample_min_size=256,):
        """
        Encode input video/image to latent space.
        
        Args:
            x: Input tensor (B, C, T, H, W) or (B, C, H, W)
            sample: Whether to sample from the posterior distribution
            is_init_image: Whether this is the initial image for chunked processing
            temporal_chunk: Whether to use temporal chunking
            window_size: Window size for chunked processing
            tile_sample_min_size: Minimum tile size for tiled processing
            
        Returns:
            Latent tensor
        """
        B = x.shape[0]
        xdim = x.ndim

        if xdim == 4:
            # The input is an image, add temporal dimension
            x = x.unsqueeze(2)

        # Use the VideoVAE_ encode method with proper scale
        if sample:
            # For sampling, we need both mu and log_var
            # Replicate the encode logic to get both
            self.vae.clear_cache()
            t = x.shape[2]
            iter_ = 1 + (t - 1) // 4

            # Encode with chunking
            for i in range(iter_):
                self.vae._enc_conv_idx = [0]
                if i == 0:
                    out = self.vae.encoder(x[:, :, :1, :, :],
                                       feat_cache=self.vae._enc_feat_map,
                                       feat_idx=self.vae._enc_conv_idx)
                else:
                    out_ = self.vae.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                        feat_cache=self.vae._enc_feat_map,
                                        feat_idx=self.vae._enc_conv_idx)
                    out = torch.cat([out, out_], 2)
            
            # Get mu and log_var from conv1
            mu, log_var = self.vae.conv1(out).chunk(2, dim=1)
            
            # Apply scale normalization to mu (like in VideoVAE_.encode)
            scale_mean = self.scale_mean.to(dtype=mu.dtype, device=mu.device)
            scale_inv_std = (1.0 / self.scale_std).to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale_mean.view(1, self.vae.z_dim, 1, 1, 1)) * scale_inv_std.view(1, self.vae.z_dim, 1, 1, 1)
            
            # Sample from posterior
            latent = self.vae.reparameterize(mu, log_var)
        else:
            # For deterministic encoding, just get mu (mode)
            latent = self.vae.encode(x, scale=self.training_scale)

        return latent

    def decode(self, latent, is_init_image=True, temporal_chunk=False, 
            window_size=2, tile_sample_min_size=256,):
        """
        Decode latent tensor back to video/image space.
        
        Args:
            latent: Latent tensor (B, C, T, H, W) or (B, C, H, W)
            is_init_image: Whether this is the initial image for chunked processing
            temporal_chunk: Whether to use temporal chunking
            window_size: Window size for chunked processing
            tile_sample_min_size: Minimum tile size for tiled processing
            
        Returns:
            Decoded video/image tensor
        """
        B = latent.shape[0]
        xdim = latent.ndim

        if xdim == 4:
            # The input is an image latent, add temporal dimension
            latent = latent.unsqueeze(2)

        # Use the VideoVAE_ decode method with proper scale
        decoded = self.vae.decode(latent, scale=self.training_scale)

        return decoded

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def reconstruct(
        self, x, sample=False, return_latent=False, is_init_image=True, 
        temporal_chunk=False, window_size=16, tile_sample_min_size=256, **kwargs
    ):
        """
        Full reconstruction pipeline: encode -> decode.
        
        Args:
            x: Input tensor
            sample: Whether to sample from posterior during encoding
            return_latent: Whether to return the latent as well
            is_init_image: Whether this is initial image for chunked processing
            temporal_chunk: Whether to use temporal chunking
            window_size: Window size for chunked processing
            tile_sample_min_size: Minimum tile size for tiled processing
            
        Returns:
            List of PIL images, optionally with latent tensor
        """
        assert x.shape[0] == 1, "Reconstruction currently supports batch size 1"
        
        # For WanVideoVAE, we don't have the same window size scaling as CausalVideoVAE
        # The VideoVAE_ handles chunking internally
        
        # Encode
        latent = self.encode(
            x, sample, is_init_image, temporal_chunk, window_size, tile_sample_min_size,
        )
        encode_latent = latent

        # Decode  
        x_recon = self.decode(
            latent, is_init_image, temporal_chunk, window_size, tile_sample_min_size
        )
        
        # Post-process to [0, 1] range
        output_image = x_recon.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)

        # Convert to PIL images
        output_image = rearrange(output_image, "B C T H W -> (B T) C H W")
        output_image = output_image.cpu().detach().permute(0, 2, 3, 1).numpy()
        output_images = self.numpy_to_pil(output_image)

        if return_latent:
            return output_images, encode_latent
        
        return output_images

    def encode_latent(self, x, sample=False, is_init_image=True, 
            temporal_chunk=False, window_size=16, tile_sample_min_size=256,):
        """
        Alias for encode method.
        """
        return self.encode(
            x, sample, is_init_image, temporal_chunk, window_size, tile_sample_min_size,
        )

    def decode_latent(self, latent, is_init_image=True, 
        temporal_chunk=False, window_size=2, tile_sample_min_size=256,):
        """
        Decode latent and convert to PIL images.
        """
        x_recon = self.decode(
            latent, is_init_image, temporal_chunk, window_size, tile_sample_min_size
        )
        
        # Post-process to [0, 1] range  
        output_image = x_recon.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        
        # Convert to PIL images
        output_image = rearrange(output_image, "B C T H W -> (B T) C H W")
        output_image = output_image.cpu().detach().permute(0, 2, 3, 1).numpy()
        output_images = self.numpy_to_pil(output_image)
        return output_images

