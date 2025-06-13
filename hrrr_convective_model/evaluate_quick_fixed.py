#!/usr/bin/env python3
"""
Quick evaluation of DDPM model - fixed version.
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule

def quick_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = UltraMinimalDDPM(in_channels=7, out_channels=7, base_dim=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(timesteps=1000, s=0.008)
    
    # Move schedule tensors to device
    noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
    noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
    noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
    noise_schedule.betas = noise_schedule.betas.to(device)
    noise_schedule.alphas = noise_schedule.alphas.to(device)
    
    # Generate small ensemble with proper DDPM sampling
    print("\nGenerating 5 ensemble members...")
    ensemble = []
    
    # Use smaller resolution for quick test
    H, W = 265, 450  # 1/4 of full resolution
    
    for i in range(5):
        print(f"  Member {i+1}/5...")
        # Start from noise
        x = torch.randn(1, 7, H, W, device=device)
        
        # Proper DDPM reverse process
        with torch.no_grad():
            # Sample fewer timesteps for speed
            timesteps = list(range(0, 1000, 20))[::-1]  # Every 20th timestep, reversed
            
            for t_idx in timesteps:
                t = torch.tensor([t_idx], device=device)
                
                # Predict noise
                noise_pred = model(x, t)
                
                # Get coefficients
                alpha = noise_schedule.alphas[t_idx]
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                beta = noise_schedule.betas[t_idx]
                
                if t_idx > 0:
                    # Standard DDPM reverse step
                    alpha_bar_prev = noise_schedule.alphas_cumprod[t_idx - 1]
                    
                    # Predict x0
                    x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    
                    # Clip to reasonable range (assuming normalized data)
                    x0_pred = torch.clamp(x0_pred, -3, 3)
                    
                    # Compute posterior mean
                    posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                   (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                    
                    # Add noise
                    posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    noise = torch.randn_like(x) if t_idx > 1 else 0
                    x = posterior_mean + posterior_var.sqrt() * noise
                else:
                    # Final step
                    x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
        
        ensemble.append(x.cpu())
    
    # Stack ensemble
    ensemble = torch.cat(ensemble, dim=0)
    
    # Compute statistics
    ens_mean = ensemble.mean(dim=0, keepdim=True)
    ens_std = ensemble.std(dim=0, keepdim=True)
    
    print(f"\nEnsemble statistics:")
    print(f"  Mean spread: {ens_std.mean().item():.4f}")
    print(f"  Max spread: {ens_std.max().item():.4f}")
    print(f"  Min spread: {ens_std.min().item():.4f}")
    
    # Quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot first variable (REFC)
    im1 = axes[0, 0].imshow(ens_mean[0, 0].numpy(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('REFC - Ensemble Mean')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(ens_std[0, 0].numpy(), cmap='plasma', aspect='auto')
    axes[0, 1].set_title('REFC - Ensemble Spread')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Plot CAPE
    im3 = axes[1, 0].imshow(ens_mean[0, 5].numpy(), cmap='viridis', aspect='auto')
    axes[1, 0].set_title('CAPE - Ensemble Mean')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(ens_std[0, 5].numpy(), cmap='plasma', aspect='auto')
    axes[1, 1].set_title('CAPE - Ensemble Spread')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('quick_ensemble_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to quick_ensemble_test.png")
    
    # Check sample quality
    print(f"\nSample quality check:")
    print(f"  Generated values range: [{ensemble.min().item():.2f}, {ensemble.max().item():.2f}]")
    print(f"  Mean: {ensemble.mean().item():.3f}")
    print(f"  Std: {ensemble.std().item():.3f}")
    
    # Per-variable statistics
    var_names = ['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN']
    print(f"\nPer-variable spread:")
    for i, var in enumerate(var_names):
        var_spread = ens_std[0, i].mean().item()
        print(f"  {var}: {var_spread:.4f}")
    
    print("\n✓ Quick evaluation complete!")

if __name__ == '__main__':
    quick_evaluate()