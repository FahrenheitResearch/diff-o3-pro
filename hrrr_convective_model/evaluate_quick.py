#!/usr/bin/env python3
"""
Quick evaluation of DDPM model with fewer samples for faster results.
"""

import torch
import numpy as np
from pathlib import Path
import json
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
    noise_schedule = CosineBetaSchedule(timesteps=1000)
    
    # Generate small ensemble quickly
    print("\nGenerating 5 ensemble members with 20 diffusion steps...")
    ensemble = []
    
    for i in range(5):
        print(f"  Member {i+1}/5...")
        # Start from noise (reduced resolution for quick test)
        x = torch.randn(1, 7, 265, 450, device=device)  # 1/4 resolution
        
        # Faster sampling - fewer steps
        with torch.no_grad():
            for t in range(20, 0, -1):  # Only 20 steps
                t_idx = t * 50 - 1  # Sample every 50th timestep
                t_tensor = torch.tensor([t_idx], device=device)
                
                # Predict noise
                noise_pred = model(x, t_tensor)
                
                # Simplified denoising
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
                
                if t > 1:
                    # Add small noise
                    x += 0.1 * torch.randn_like(x)
        
        ensemble.append(x.cpu())
    
    # Stack ensemble
    ensemble = torch.cat(ensemble, dim=0)
    
    # Compute statistics
    ens_mean = ensemble.mean(dim=0)
    ens_std = ensemble.std(dim=0)
    
    print(f"\nEnsemble statistics:")
    print(f"  Mean spread: {ens_std.mean().item():.4f}")
    print(f"  Max spread: {ens_std.max().item():.4f}")
    print(f"  Min spread: {ens_std.min().item():.4f}")
    
    # Quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot first variable (REFC)
    axes[0, 0].imshow(ens_mean[0, 0].numpy(), cmap='viridis')
    axes[0, 0].set_title('REFC - Ensemble Mean')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ens_std[0, 0].numpy(), cmap='plasma')
    axes[0, 1].set_title('REFC - Ensemble Spread')
    axes[0, 1].axis('off')
    
    # Plot CAPE
    axes[1, 0].imshow(ens_mean[0, 5].numpy(), cmap='viridis')
    axes[1, 0].set_title('CAPE - Ensemble Mean')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ens_std[0, 5].numpy(), cmap='plasma')
    axes[1, 1].set_title('CAPE - Ensemble Spread')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_ensemble_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to quick_ensemble_test.png")
    
    # Check sample quality
    print(f"\nSample quality check:")
    print(f"  Generated values range: [{ensemble.min().item():.2f}, {ensemble.max().item():.2f}]")
    print(f"  Mean: {ensemble.mean().item():.3f}")
    print(f"  Std: {ensemble.std().item():.3f}")
    
    print("\n✓ Quick evaluation complete!")

if __name__ == '__main__':
    quick_evaluate()