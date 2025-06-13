#!/usr/bin/env python3
"""
Test conditional forecasting - verify the model actually uses current conditions.
This is the moment of truth - does our model forecast or just generate random weather?
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from models.diffusion.ddpm_conditional_minimal import ConditionalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def test_conditional_model():
    """Test if the conditional model actually conditions on current state."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*70)
    print("TESTING CONDITIONAL DIFFUSION MODEL")
    print("This will verify if forecasts depend on current conditions")
    print("="*70 + "\n")
    
    # Load checkpoint
    checkpoint_dir = Path('checkpoints/conditional_faithful')
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    
    if not checkpoint_path.exists():
        # Try latest epoch
        checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            print("ERROR: No checkpoint found! Train the model first.")
            return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = ConditionalDDPM(
        channels=config['model']['channels'],
        base_dim=config['model']['base_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion']['s']
    )
    # Move schedule to device
    for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Load dataset
    normalizer = Normalizer(config['data']['stats'])
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Get a sample with interesting weather
    print("\nFinding sample with convection...")
    for idx in range(100, 500):
        current_state, future_true = dataset[idx]
        # Check if there's significant REFC (convection)
        refc_denorm = normalizer.decode(current_state[0].numpy(), 'REFC')
        if refc_denorm.max() > 20:  # Found convection
            print(f"Using sample {idx} with max REFC = {refc_denorm.max():.1f} dBZ")
            break
    
    current_state = current_state.unsqueeze(0).to(device)
    future_true = future_true.unsqueeze(0).to(device)
    
    # Generate ensemble forecast
    print("\nGenerating 5-member ensemble...")
    ensemble_members = []
    
    with torch.no_grad():
        for member in range(5):
            print(f"\nMember {member+1}:")
            
            # Start from pure noise
            x = torch.randn_like(future_true)
            
            # DDPM reverse process - CONDITIONED on current state!
            timesteps = list(range(999, -1, -20))  # Every 20th step for speed
            
            for t_idx in tqdm(timesteps, desc="Denoising"):
                t = torch.tensor([t_idx], device=device)
                
                # CRITICAL: Pass current_state as condition
                noise_pred = model(x, t, current_state)
                
                # DDPM denoising step
                alpha = noise_schedule.alphas[t_idx]
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                beta = noise_schedule.betas[t_idx]
                
                if t_idx > 0:
                    # Not the last step
                    alpha_bar_prev = noise_schedule.alphas_cumprod[t_idx - 1]
                    
                    # Predict x0
                    x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x0_pred = torch.clamp(x0_pred, -3, 3)
                    
                    # Compute posterior mean and variance
                    posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                   (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                    posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    
                    # Add noise (except for t=0)
                    noise = torch.randn_like(x) if t_idx > 0 else 0
                    x = posterior_mean + posterior_var.sqrt() * noise
                else:
                    # Last step - no noise
                    x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
            
            ensemble_members.append(x.cpu())
    
    # Stack ensemble
    ensemble = torch.cat(ensemble_members, dim=0)
    
    # Denormalize for visualization
    print("\nDenormalizing and creating visualizations...")
    var_names = config['data']['variables']
    
    # Focus on REFC (convection)
    refc_idx = var_names.index('REFC')
    
    # Denormalize
    current_denorm = normalizer.decode(current_state.cpu()[0, refc_idx].numpy(), 'REFC')
    future_true_denorm = normalizer.decode(future_true.cpu()[0, refc_idx].numpy(), 'REFC')
    ensemble_denorm = np.array([
        normalizer.decode(ensemble[i, refc_idx].numpy(), 'REFC') 
        for i in range(5)
    ])
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Current state
    im1 = axes[0, 0].imshow(current_denorm, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
    axes[0, 0].set_title('Current State (REFC)', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='dBZ')
    
    # True future
    im2 = axes[0, 1].imshow(future_true_denorm, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
    axes[0, 1].set_title('True Future (+1h)', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='dBZ')
    
    # Ensemble mean
    ens_mean = ensemble_denorm.mean(axis=0)
    im3 = axes[0, 2].imshow(ens_mean, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
    axes[0, 2].set_title('Ensemble Mean Forecast', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='dBZ')
    
    # Individual members
    for i in range(3):
        im = axes[1, i].imshow(ensemble_denorm[i], cmap='turbo', vmin=-10, vmax=60, aspect='auto')
        axes[1, i].set_title(f'Member {i+1}', fontsize=12)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], label='dBZ')
    
    plt.suptitle('CONDITIONAL Diffusion Model Test - Does it use current conditions?', fontsize=16)
    plt.tight_layout()
    
    # Save
    output_path = Path('forecasts') / f'conditional_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved test results to: {output_path}")
    
    # Compute basic metrics
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"Current state - Max REFC: {current_denorm.max():.1f} dBZ")
    print(f"True future   - Max REFC: {future_true_denorm.max():.1f} dBZ")
    print(f"Forecast mean - Max REFC: {ens_mean.max():.1f} dBZ")
    print(f"Ensemble spread (std): {ensemble_denorm.std(axis=0).mean():.2f} dBZ")
    
    # Check if forecast is correlated with current state
    from scipy.stats import pearsonr
    corr, _ = pearsonr(current_denorm.flatten(), ens_mean.flatten())
    print(f"\nCorrelation between current and forecast: {corr:.3f}")
    
    if corr > 0.3:
        print("✓ SUCCESS: Forecast is correlated with current conditions!")
        print("The model IS using the current state to make predictions.")
    else:
        print("✗ FAILURE: Forecast seems independent of current conditions.")
        print("The model might not be properly conditioned.")
    
    print("=" * 70)
    
    # Also save individual PNGs for closer inspection
    individual_dir = Path('forecasts') / f'conditional_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}_individual'
    individual_dir.mkdir(exist_ok=True)
    
    # Save current
    plt.figure(figsize=(10, 8))
    plt.imshow(current_denorm, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
    plt.colorbar(label='dBZ')
    plt.title('REFC - Current State')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(individual_dir / 'current_state.png', dpi=150)
    plt.close()
    
    # Save ensemble mean
    plt.figure(figsize=(10, 8))
    plt.imshow(ens_mean, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
    plt.colorbar(label='dBZ')
    plt.title('REFC - Ensemble Mean (+1h)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(individual_dir / 'ensemble_mean.png', dpi=150)
    plt.close()
    
    print(f"\nIndividual PNGs saved to: {individual_dir}")


if __name__ == '__main__':
    test_conditional_model()