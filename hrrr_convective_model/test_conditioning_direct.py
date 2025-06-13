#!/usr/bin/env python3
"""
Direct test of conditioning - generate forecasts from same initial condition
to verify reproducibility and conditioning.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from models.diffusion.ddpm_conditional_minimal import ConditionalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def test_direct_conditioning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDirect Conditioning Test")
    print("========================\n")
    
    # Load model
    checkpoint = torch.load('checkpoints/conditional_faithful/best_model.pt', 
                          map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = ConditionalDDPM(
        channels=config['model']['channels'],
        base_dim=config['model']['base_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup
    noise_schedule = CosineBetaSchedule(timesteps=1000, s=0.008)
    for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    normalizer = Normalizer(config['data']['stats'])
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Get two different weather states
    print("Loading two different initial conditions...")
    idx1, idx2 = 100, 200
    state1, _ = dataset[idx1]
    state2, _ = dataset[idx2]
    
    state1 = state1.unsqueeze(0).to(device)
    state2 = state2.unsqueeze(0).to(device)
    
    # Test 1: Same initial noise, different conditions
    print("\nTest 1: Same noise, different initial conditions")
    torch.manual_seed(42)
    initial_noise = torch.randn_like(state1)
    
    # Forecast from state1
    x1 = initial_noise.clone()
    with torch.no_grad():
        for t in [999, 799, 599, 399, 199, 99, 49, 19, 0]:
            t_tensor = torch.tensor([t], device=device)
            noise_pred = model(x1, t_tensor, state1)
            
            # Denoising step
            alpha_bar = noise_schedule.alphas_cumprod[t]
            if t > 0:
                alpha_bar_prev = noise_schedule.alphas_cumprod[t - 1]
                beta = noise_schedule.betas[t]
                
                # Predict x0
                x0_pred = (x1 - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x0_pred = torch.clamp(x0_pred, -3, 3)
                
                # Sample x_{t-1}
                mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                       ((1 - alpha_bar_prev) * (1 - beta).sqrt() / (1 - alpha_bar)) * x1
                       
                if t > 1:
                    var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    noise = torch.randn_like(x1) * var.sqrt()
                    x1 = mean + noise
                else:
                    x1 = mean
            else:
                x1 = (x1 - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
    
    forecast1 = x1.cpu()
    
    # Forecast from state2  
    x2 = initial_noise.clone()
    with torch.no_grad():
        for t in [999, 799, 599, 399, 199, 99, 49, 19, 0]:
            t_tensor = torch.tensor([t], device=device)
            noise_pred = model(x2, t_tensor, state2)
            
            # Same denoising
            alpha_bar = noise_schedule.alphas_cumprod[t]
            if t > 0:
                alpha_bar_prev = noise_schedule.alphas_cumprod[t - 1]
                beta = noise_schedule.betas[t]
                
                x0_pred = (x2 - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x0_pred = torch.clamp(x0_pred, -3, 3)
                
                mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                       ((1 - alpha_bar_prev) * (1 - beta).sqrt() / (1 - alpha_bar)) * x2
                       
                if t > 1:
                    var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    noise = torch.randn_like(x2) * var.sqrt()
                    x2 = mean + noise
                else:
                    x2 = mean
            else:
                x2 = (x2 - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
    
    forecast2 = x2.cpu()
    
    # Compare
    diff = (forecast1 - forecast2).abs().mean().item()
    print(f"Mean absolute difference between forecasts: {diff:.4f}")
    
    # Visualize REFC channel
    refc_idx = 0
    
    # Denormalize
    state1_denorm = normalizer.decode(state1.cpu()[0, refc_idx].numpy(), 'REFC')
    state2_denorm = normalizer.decode(state2.cpu()[0, refc_idx].numpy(), 'REFC')
    forecast1_denorm = normalizer.decode(forecast1[0, refc_idx].numpy(), 'REFC')
    forecast2_denorm = normalizer.decode(forecast2[0, refc_idx].numpy(), 'REFC')
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initial conditions
    im1 = axes[0, 0].imshow(state1_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[0, 0].set_title('Initial Condition 1')
    plt.colorbar(im1, ax=axes[0, 0], label='dBZ')
    
    im2 = axes[0, 1].imshow(state2_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[0, 1].set_title('Initial Condition 2')
    plt.colorbar(im2, ax=axes[0, 1], label='dBZ')
    
    # Forecasts
    im3 = axes[1, 0].imshow(forecast1_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[1, 0].set_title('Forecast from Condition 1')
    plt.colorbar(im3, ax=axes[1, 0], label='dBZ')
    
    im4 = axes[1, 1].imshow(forecast2_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[1, 1].set_title('Forecast from Condition 2')
    plt.colorbar(im4, ax=axes[1, 1], label='dBZ')
    
    plt.suptitle('Conditioning Test: Same Initial Noise, Different Conditions', fontsize=14)
    plt.tight_layout()
    plt.savefig('forecasts/direct_conditioning_test.png', dpi=150)
    print("\nSaved visualization to: forecasts/direct_conditioning_test.png")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Initial state 1 max REFC: {state1_denorm.max():.1f} dBZ")
    print(f"Initial state 2 max REFC: {state2_denorm.max():.1f} dBZ")
    print(f"Forecast 1 max REFC: {forecast1_denorm.max():.1f} dBZ")
    print(f"Forecast 2 max REFC: {forecast2_denorm.max():.1f} dBZ")
    print(f"\nDifference in forecasts: {diff:.4f}")
    
    if diff > 0.1:
        print("✓ Model IS using conditioning (forecasts differ)")
    else:
        print("✗ Model NOT using conditioning properly")
    print("="*60)


if __name__ == '__main__':
    test_direct_conditioning()