#!/usr/bin/env python3
"""
Script to plot training metrics from detailed_metrics.json
Usage: python plot_training_metrics.py --metrics_file /path/to/detailed_metrics.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_metrics(metrics_file, save_dir=None):
    """Plot training and evaluation metrics from JSON file."""
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No data found in metrics file")
        return
    
    # Extract data
    epochs = [entry['epoch'] for entry in data]
    
    # Prepare subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VAE Training Metrics', fontsize=16)
    
    # Loss metrics
    train_vae_loss = [entry.get('train_vae_loss', 0) for entry in data]
    eval_vae_loss = [entry.get('eval_vae_loss', 0) for entry in data]
    
    axes[0, 0].plot(epochs, train_vae_loss, 'b-', label='Train VAE Loss', alpha=0.7)
    if any(eval_vae_loss):
        eval_epochs = [epochs[i] for i, val in enumerate(eval_vae_loss) if val > 0]
        eval_vals = [val for val in eval_vae_loss if val > 0]
        axes[0, 0].plot(eval_epochs, eval_vals, 'r-', label='Eval VAE Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('VAE Loss')
    axes[0, 0].set_title('VAE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # GAN loss
    train_disc_loss = [entry.get('train_disc_loss', 0) for entry in data]
    eval_gan_loss = [entry.get('eval_gan_loss', 0) for entry in data]
    
    axes[0, 1].plot(epochs, train_disc_loss, 'b-', label='Train Disc Loss', alpha=0.7)
    if any(eval_gan_loss):
        eval_epochs = [epochs[i] for i, val in enumerate(eval_gan_loss) if val > 0]
        eval_vals = [val for val in eval_gan_loss if val > 0]
        axes[0, 1].plot(eval_epochs, eval_vals, 'r-', label='Eval GAN Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('GAN Loss')
    axes[0, 1].set_title('GAN Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rates
    lr = [entry.get('train_lr', 0) for entry in data]
    disc_lr = [entry.get('train_disc_lr', 0) for entry in data]
    
    axes[0, 2].plot(epochs, lr, 'g-', label='VAE LR', alpha=0.7)
    if any(disc_lr):
        axes[0, 2].plot(epochs, disc_lr, 'm-', label='Disc LR', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rates')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Detailed loss components
    kl_loss = [entry.get('train_kl_loss', 0) for entry in data]
    perceptual_loss = [entry.get('train_perceptual_loss', 0) for entry in data]
    
    axes[1, 0].plot(epochs, kl_loss, 'orange', label='KL Loss', alpha=0.7)
    axes[1, 0].plot(epochs, perceptual_loss, 'purple', label='Perceptual Loss', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction quality metrics
    psnr = [entry.get('reconstruction_psnr', 0) for entry in data]
    ssim = [entry.get('reconstruction_ssim', 0) for entry in data]
    
    if any(psnr) or any(ssim):
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        if any(psnr):
            psnr_epochs = [epochs[i] for i, val in enumerate(psnr) if val > 0]
            psnr_vals = [val for val in psnr if val > 0]
            line1 = ax1.plot(psnr_epochs, psnr_vals, 'b-', label='PSNR', alpha=0.7)
            ax1.set_ylabel('PSNR (dB)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
        
        if any(ssim):
            ssim_epochs = [epochs[i] for i, val in enumerate(ssim) if val > 0]
            ssim_vals = [val for val in ssim if val > 0]
            line2 = ax2.plot(ssim_epochs, ssim_vals, 'r-', label='SSIM', alpha=0.7)
            ax2.set_ylabel('SSIM', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.set_xlabel('Epoch')
        ax1.set_title('Reconstruction Quality')
        ax1.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No reconstruction\nquality metrics', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Reconstruction Quality')
    
    # Training progress overview
    total_loss = [entry.get('train_vae_loss', 0) + entry.get('train_disc_loss', 0) for entry in data]
    grad_norm = [entry.get('train_grad_norm', 0) for entry in data]
    
    ax1 = axes[1, 2]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, total_loss, 'g-', label='Total Loss', alpha=0.7)
    ax1.set_ylabel('Total Loss', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    
    if any(grad_norm):
        line2 = ax2.plot(epochs, grad_norm, 'orange', label='Grad Norm', alpha=0.7)
        ax2.set_ylabel('Gradient Norm', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot VAE training metrics')
    parser.add_argument('--metrics_file', required=True, help='Path to detailed_metrics.json')
    parser.add_argument('--save_dir', help='Directory to save plots (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.metrics_file).exists():
        print(f"Metrics file not found: {args.metrics_file}")
        return
    
    plot_metrics(args.metrics_file, args.save_dir)


if __name__ == '__main__':
    main() 