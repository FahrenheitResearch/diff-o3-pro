#!/usr/bin/env python3
"""
Generate ensemble forecast focused on convective potential.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def generate_convective_forecast():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = UltraMinimalDDPM(in_channels=7, out_channels=7, base_dim=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    config = checkpoint['config']
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(timesteps=1000, s=0.008)
    for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Load data
    normalizer = Normalizer(config['data']['stats'])
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Find high convective potential case
    print("Searching for convective case...")
    best_idx = 0
    max_score = 0
    
    for i in range(min(200, len(dataset))):
        current, _ = dataset[i]
        # Score based on CAPE and CIN
        cape = normalizer.decode(current[5].numpy(), 'CAPE')
        cin = normalizer.decode(current[6].numpy(), 'CIN')
        
        # High CAPE, moderate CIN = good convective potential
        score = cape.max() - 0.5 * np.abs(cin.min())
        if score > max_score:
            max_score = score
            best_idx = i
    
    print(f"Selected case with convective score: {max_score:.0f}")
    current_state, _ = dataset[best_idx]
    current_state = current_state.unsqueeze(0).to(device)
    
    # Generate focused ensemble
    num_members = 15
    ensemble_members = []
    
    print(f"\nGenerating {num_members}-member convective ensemble...")
    
    for i in tqdm(range(num_members)):
        x = torch.randn_like(current_state)
        
        with torch.no_grad():
            # Fewer steps for speed
            timesteps = np.linspace(999, 0, 40, dtype=int)
            
            for t_idx in timesteps:
                t = torch.tensor([t_idx], device=device)
                noise_pred = model(x, t)
                
                alpha = noise_schedule.alphas[t_idx]
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                beta = noise_schedule.betas[t_idx]
                
                if t_idx > 0:
                    alpha_bar_prev = noise_schedule.alphas_cumprod[t_idx - 1]
                    x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x0_pred = torch.clamp(x0_pred, -3, 3)
                    
                    posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                   (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                    posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    
                    noise = torch.randn_like(x) if t_idx > 1 else 0
                    x = posterior_mean + posterior_var.sqrt() * noise
                else:
                    x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
        
        ensemble_members.append(x.cpu())
    
    ensemble = torch.cat(ensemble_members, dim=0)
    
    # Denormalize
    var_names = config['data']['variables']
    ensemble_denorm = torch.zeros_like(ensemble)
    current_denorm = torch.zeros_like(current_state.cpu())
    
    for i, var in enumerate(var_names):
        for m in range(num_members):
            ensemble_denorm[m, i] = torch.tensor(normalizer.decode(ensemble[m, i].numpy(), var))
        current_denorm[0, i] = torch.tensor(normalizer.decode(current_state.cpu()[0, i].numpy(), var))
    
    # Create convection-focused visualization
    print("\nCreating convection analysis...")
    fig = plt.figure(figsize=(20, 16))
    
    # Define convective thresholds
    cape_thresh = [500, 1000, 2000, 3000]
    cin_thresh = [-200, -100, -50, -25]
    refc_thresh = [20, 35, 45]
    
    # 1. CAPE probability cascades
    for i, thresh in enumerate(cape_thresh):
        ax = plt.subplot(4, 4, i+1)
        prob = (ensemble_denorm[:, 5] > thresh).float().mean(dim=0)
        im = ax.imshow(prob * 100, cmap='hot_r', vmin=0, vmax=100, aspect='auto')
        ax.set_title(f'P(CAPE > {thresh} J/kg)', fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('%')
    
    # 2. CIN analysis
    for i, thresh in enumerate(cin_thresh):
        ax = plt.subplot(4, 4, i+5)
        prob = (ensemble_denorm[:, 6] > thresh).float().mean(dim=0)
        im = ax.imshow(prob * 100, cmap='RdBu_r', vmin=0, vmax=100, aspect='auto')
        ax.set_title(f'P(CIN > {thresh} J/kg)', fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('%')
    
    # 3. Composite Convective Threat
    ax = plt.subplot(4, 4, 9)
    # Threat = high CAPE, moderate CIN, significant reflectivity potential
    cape_threat = (ensemble_denorm[:, 5] > 1500).float().mean(dim=0)
    cin_favorable = ((ensemble_denorm[:, 6] > -150) & (ensemble_denorm[:, 6] < -25)).float().mean(dim=0)
    refc_potential = (ensemble_denorm[:, 0] > 30).float().mean(dim=0)
    
    composite_threat = (cape_threat + cin_favorable + refc_potential) / 3
    im = ax.imshow(composite_threat * 100, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax.set_title('Composite Convective Threat (%)', fontsize=11, weight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. Maximum values in ensemble
    ax = plt.subplot(4, 4, 10)
    max_cape = ensemble_denorm[:, 5].max(dim=0)[0]
    im = ax.imshow(max_cape, cmap='plasma', vmin=0, vmax=4000, aspect='auto')
    ax.set_title('Maximum CAPE in Ensemble', fontsize=10)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('J/kg')
    
    # 5. Reflectivity scenarios
    percentiles = [10, 50, 90]
    for i, p in enumerate(percentiles):
        ax = plt.subplot(4, 4, 11+i)
        refc_percentile = torch.quantile(ensemble_denorm[:, 0], p/100, dim=0)
        im = ax.imshow(refc_percentile, cmap='turbo', vmin=-10, vmax=60, aspect='auto')
        ax.set_title(f'REFC {p}th Percentile', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 6. Uncertainty vs Intensity plot
    ax = plt.subplot(4, 4, 14)
    cape_mean = ensemble_denorm[:, 5].mean(dim=0)
    cape_std = ensemble_denorm[:, 5].std(dim=0)
    
    # Scatter plot (subsample for clarity)
    stride = 50
    x = cape_mean[::stride, ::stride].flatten()
    y = cape_std[::stride, ::stride].flatten()
    
    scatter = ax.scatter(x, y, c=x, cmap='viridis', alpha=0.5, s=1)
    ax.set_xlabel('Mean CAPE (J/kg)')
    ax.set_ylabel('CAPE Spread (J/kg)')
    ax.set_title('Uncertainty vs Intensity')
    ax.grid(True, alpha=0.3)
    
    # 7. Convective regions
    ax = plt.subplot(4, 4, 15)
    # Define convective regions
    high_prob = composite_threat > 0.5
    moderate_prob = (composite_threat > 0.25) & (composite_threat <= 0.5)
    low_prob = (composite_threat > 0.1) & (composite_threat <= 0.25)
    
    # Create color map
    region_map = torch.zeros_like(composite_threat)
    region_map[low_prob] = 1
    region_map[moderate_prob] = 2
    region_map[high_prob] = 3
    
    im = ax.imshow(region_map, cmap='RdYlGn_r', vmin=0, vmax=3, aspect='auto')
    ax.set_title('Convective Risk Categories', fontsize=11, weight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', label='None'),
        Patch(facecolor='yellow', label='Low'),
        Patch(facecolor='orange', label='Moderate'),
        Patch(facecolor='red', label='High')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 8. Summary statistics box
    ax = plt.subplot(4, 4, 16)
    ax.axis('off')
    
    # Calculate domain statistics
    domain_stats = f"""DOMAIN CONVECTIVE STATISTICS
    
Maximum Values:
  CAPE: {ensemble_denorm[:, 5].max().item():.0f} J/kg
  REFC: {ensemble_denorm[:, 0].max().item():.1f} dBZ
  CIN: {ensemble_denorm[:, 6].max().item():.0f} J/kg

Coverage (% of domain):
  CAPE > 1000: {(ensemble_denorm[:, 5] > 1000).any(dim=0).float().mean().item()*100:.1f}%
  CAPE > 2000: {(ensemble_denorm[:, 5] > 2000).any(dim=0).float().mean().item()*100:.1f}%
  Favorable CIN: {((ensemble_denorm[:, 6] > -150) & (ensemble_denorm[:, 6] < -25)).any(dim=0).float().mean().item()*100:.1f}%
  
Threat Areas:
  High: {(composite_threat > 0.5).float().mean().item()*100:.1f}%
  Moderate: {((composite_threat > 0.25) & (composite_threat <= 0.5)).float().mean().item()*100:.1f}%
  Low: {((composite_threat > 0.1) & (composite_threat <= 0.25)).float().mean().item()*100:.1f}%"""
    
    ax.text(0.05, 0.95, domain_stats, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Convective Potential Analysis - {num_members} Member Ensemble\n'
                 f'Valid: {datetime.now() + timedelta(hours=1):%Y-%m-%d %H:00 UTC} (+1h)',
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('forecasts')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'convective_forecast_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved convective analysis to: {plot_file}")
    print("\n✓ Convective forecast complete!")


if __name__ == '__main__':
    generate_convective_forecast()