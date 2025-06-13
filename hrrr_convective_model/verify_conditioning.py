#!/usr/bin/env python3
"""
Verify if the model is actually conditioning on input or just generating noise.
We'll test:
1. Generate multiple forecasts from SAME initial condition - should be similar
2. Generate forecasts from DIFFERENT conditions - should be different
3. Check if the model just passes through the input (no learning)
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import pearsonr

from models.diffusion.ddpm_conditional_minimal import ConditionalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def verify_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nModel Verification Test")
    print("======================\n")
    
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
    
    # Get a state with strong signal
    print("Finding sample with strong convection...")
    for idx in range(100, 300):
        current, future = dataset[idx]
        refc = normalizer.decode(current[0].numpy(), 'REFC')
        if refc.max() > 50:  # Strong convection
            print(f"Using sample {idx} with max REFC = {refc.max():.1f} dBZ")
            break
    
    current = current.unsqueeze(0).to(device)
    future_true = future.unsqueeze(0).to(device)
    
    # Test 1: Multiple samples from same condition
    print("\nTest 1: Generating 3 forecasts from SAME initial condition...")
    forecasts_same = []
    
    for i in range(3):
        torch.manual_seed(i)  # Different random seeds
        x = torch.randn_like(current)
        
        with torch.no_grad():
            # Fast sampling - fewer steps
            for t in range(999, -1, -50):
                t_tensor = torch.tensor([t], device=device)
                noise_pred = model(x, t_tensor, current)
                
                alpha_bar = noise_schedule.alphas_cumprod[t]
                if t > 0:
                    # Simplified DDPM sampling
                    x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
                    
                    # Add noise for next step
                    if t > 50:
                        beta = noise_schedule.betas[t]
                        x = x + beta.sqrt() * torch.randn_like(x) * 0.5
                else:
                    x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        
        forecasts_same.append(x.cpu())
    
    # Test 2: Different initial conditions
    print("\nTest 2: Generating forecast from DIFFERENT initial condition...")
    idx2 = idx + 100
    current2, _ = dataset[idx2]
    current2 = current2.unsqueeze(0).to(device)
    
    torch.manual_seed(0)  # Same seed as first forecast
    x = torch.randn_like(current2)
    
    with torch.no_grad():
        for t in range(999, -1, -50):
            t_tensor = torch.tensor([t], device=device)
            noise_pred = model(x, t_tensor, current2)
            
            alpha_bar = noise_schedule.alphas_cumprod[t]
            if t > 0:
                x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
                x = torch.clamp(x, -3, 3)
                if t > 50:
                    beta = noise_schedule.betas[t]
                    x = x + beta.sqrt() * torch.randn_like(x) * 0.5
            else:
                x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
    
    forecast_diff = x.cpu()
    
    # Analyze results
    print("\nAnalyzing results...")
    
    # 1. Check if forecasts from same condition are similar
    same_diffs = []
    for i in range(len(forecasts_same)-1):
        diff = (forecasts_same[i] - forecasts_same[i+1]).abs().mean().item()
        same_diffs.append(diff)
    avg_same_diff = np.mean(same_diffs)
    
    # 2. Check if forecast from different condition is different
    diff_cond = (forecasts_same[0] - forecast_diff).abs().mean().item()
    
    # 3. Check if model just returns input (correlation test)
    input_output_corr = pearsonr(current.cpu().flatten().numpy(), 
                                 forecasts_same[0].flatten().numpy())[0]
    
    # 4. Check signal strength
    input_range = (current.max() - current.min()).item()
    output_range = (forecasts_same[0].max() - forecasts_same[0].min()).item()
    
    # Visualize
    refc_idx = 0
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Current state
    curr_denorm = normalizer.decode(current.cpu()[0, refc_idx].numpy(), 'REFC')
    im = axes[0, 0].imshow(curr_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[0, 0].set_title(f'Input (max={curr_denorm.max():.1f} dBZ)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Three forecasts from same condition
    for i in range(3):
        fc_denorm = normalizer.decode(forecasts_same[i][0, refc_idx].numpy(), 'REFC')
        im = axes[0, i+1].imshow(fc_denorm, cmap='turbo', vmin=-10, vmax=60)
        axes[0, i+1].set_title(f'Same Cond #{i+1} (max={fc_denorm.max():.1f})')
        plt.colorbar(im, ax=axes[0, i+1])
    
    # Different condition and its forecast
    curr2_denorm = normalizer.decode(current2.cpu()[0, refc_idx].numpy(), 'REFC')
    im = axes[1, 0].imshow(curr2_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[1, 0].set_title(f'Different Input (max={curr2_denorm.max():.1f} dBZ)')
    plt.colorbar(im, ax=axes[1, 0])
    
    fc_diff_denorm = normalizer.decode(forecast_diff[0, refc_idx].numpy(), 'REFC')
    im = axes[1, 1].imshow(fc_diff_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[1, 1].set_title(f'Different Cond Forecast (max={fc_diff_denorm.max():.1f})')
    plt.colorbar(im, ax=axes[1, 1])
    
    # True future for reference
    future_denorm = normalizer.decode(future_true.cpu()[0, refc_idx].numpy(), 'REFC')
    im = axes[1, 2].imshow(future_denorm, cmap='turbo', vmin=-10, vmax=60)
    axes[1, 2].set_title(f'True Future (max={future_denorm.max():.1f})')
    plt.colorbar(im, ax=axes[1, 2])
    
    # Hide empty subplot
    axes[1, 3].axis('off')
    
    plt.suptitle('Model Verification: Is it Using Conditioning?', fontsize=16)
    plt.tight_layout()
    plt.savefig('forecasts/model_verification.png', dpi=150)
    
    # Print results
    print("\n" + "="*70)
    print("VERIFICATION RESULTS:")
    print("="*70)
    print(f"\n1. CONSISTENCY TEST (same initial condition):")
    print(f"   Average difference between forecasts: {avg_same_diff:.4f}")
    print(f"   {'✓ PASS' if avg_same_diff < 0.5 else '✗ FAIL'}: Forecasts from same condition are {'similar' if avg_same_diff < 0.5 else 'too different'}")
    
    print(f"\n2. CONDITIONING TEST (different initial conditions):")
    print(f"   Difference between forecasts: {diff_cond:.4f}")
    print(f"   Ratio to same-condition difference: {diff_cond/avg_same_diff:.2f}x")
    print(f"   {'✓ PASS' if diff_cond > avg_same_diff * 1.5 else '✗ FAIL'}: Different conditions produce {'different' if diff_cond > avg_same_diff * 1.5 else 'similar'} forecasts")
    
    print(f"\n3. TRIVIAL SOLUTION TEST (just returning input):")
    print(f"   Input-output correlation: {input_output_corr:.3f}")
    print(f"   {'✓ PASS' if abs(input_output_corr) < 0.8 else '✗ FAIL'}: Model {'is NOT' if abs(input_output_corr) < 0.8 else 'IS'} just returning input")
    
    print(f"\n4. SIGNAL STRENGTH TEST:")
    print(f"   Input range: {input_range:.3f}")
    print(f"   Output range: {output_range:.3f}")
    print(f"   Output/Input ratio: {output_range/input_range:.2f}")
    print(f"   {'✗ WEAK' if output_range/input_range < 0.3 else '✓ OK'}: Signal strength is {'very weak' if output_range/input_range < 0.3 else 'reasonable'}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    
    if avg_same_diff < 0.5 and diff_cond > avg_same_diff * 1.5 and abs(input_output_corr) < 0.8:
        print("✓ Model IS using conditioning properly")
        if output_range/input_range < 0.3:
            print("⚠ But signal is very weak - needs more training")
    else:
        print("✗ Model is NOT working properly - may be generating noise")
    
    print("=" * 70)
    print(f"\nVisualization saved to: forecasts/model_verification.png")


if __name__ == '__main__':
    verify_model()