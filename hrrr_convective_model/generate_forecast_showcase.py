#!/usr/bin/env python3
"""
Generate showcase ensemble forecast with probability maps and spaghetti plots.
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def create_probability_colormap():
    """Create custom colormap for probability plots."""
    colors = ['white', 'lightblue', 'blue', 'darkblue', 'purple', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('prob_cmap', colors, N=n_bins)
    return cmap


def generate_showcase_forecast():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    print(f"Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = UltraMinimalDDPM(
        in_channels=7,
        out_channels=7,
        base_dim=16
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    config = checkpoint['config']
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion']['s']
    )
    
    # Move to device
    for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Load normalizer
    normalizer = Normalizer(config['data']['stats'])
    
    # Load dataset
    print("Loading HRRR data...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Select interesting case
    idx = np.random.randint(100, 600)  # Middle of dataset
    current_state, _ = dataset[idx]
    current_state = current_state.unsqueeze(0).to(device)
    
    # Generate larger ensemble for better statistics
    num_members = 20
    num_steps = 50
    
    print(f"\nGenerating {num_members}-member ensemble...")
    ensemble_members = []
    
    for i in tqdm(range(num_members), desc="Ensemble generation"):
        # Start from noise
        x = torch.randn_like(current_state)
        
        # Reverse diffusion
        with torch.no_grad():
            timesteps = np.linspace(999, 0, num_steps, dtype=int)
            
            for t_idx in timesteps:
                t = torch.tensor([t_idx], device=device)
                
                # Predict noise
                noise_pred = model(x, t)
                
                # DDPM step
                alpha = noise_schedule.alphas[t_idx]
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                beta = noise_schedule.betas[t_idx]
                
                if t_idx > 0:
                    alpha_bar_prev = noise_schedule.alphas_cumprod[t_idx - 1]
                    
                    # Predict x0
                    x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x0_pred = torch.clamp(x0_pred, -3, 3)
                    
                    # Posterior
                    posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                   (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                    
                    posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    
                    noise = torch.randn_like(x) if t_idx > 1 else 0
                    x = posterior_mean + posterior_var.sqrt() * noise
                else:
                    x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
        
        ensemble_members.append(x.cpu())
    
    # Stack ensemble
    ensemble = torch.cat(ensemble_members, dim=0)
    
    # Denormalize
    print("Processing ensemble...")
    var_names = config['data']['variables']
    ensemble_denorm = torch.zeros_like(ensemble)
    current_denorm = torch.zeros_like(current_state.cpu())
    
    for i, var in enumerate(var_names):
        for m in range(num_members):
            ensemble_denorm[m, i] = torch.tensor(
                normalizer.decode(ensemble[m, i].numpy(), var)
            )
        current_denorm[0, i] = torch.tensor(
            normalizer.decode(current_state.cpu()[0, i].numpy(), var)
        )
    
    # Create comprehensive showcase visualization
    print("Creating showcase visualizations...")
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Reflectivity probability plot
    ax1 = plt.subplot(3, 3, 1)
    refc_prob_20 = (ensemble_denorm[:, 0] > 20).float().mean(dim=0)
    refc_prob_35 = (ensemble_denorm[:, 0] > 35).float().mean(dim=0)
    refc_prob_50 = (ensemble_denorm[:, 0] > 50).float().mean(dim=0)
    
    # Composite probability
    prob_composite = refc_prob_20 + refc_prob_35 + refc_prob_50
    im1 = ax1.imshow(prob_composite, cmap=create_probability_colormap(), 
                     vmin=0, vmax=3, aspect='auto')
    ax1.set_title('Reflectivity Exceedance Probability\n>20dBZ + >35dBZ + >50dBZ', fontsize=12)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Cumulative Probability')
    
    # 2. CAPE probability plot
    ax2 = plt.subplot(3, 3, 2)
    cape_prob_1000 = (ensemble_denorm[:, 5] > 1000).float().mean(dim=0)
    cape_prob_2500 = (ensemble_denorm[:, 5] > 2500).float().mean(dim=0)
    
    im2 = ax2.imshow(cape_prob_1000, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Probability of CAPE > 1000 J/kg\n(Convective Potential)', fontsize=12)
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
    cbar2.set_label('Probability')
    
    # 3. Ensemble spread normalized by mean
    ax3 = plt.subplot(3, 3, 3)
    refc_mean = ensemble_denorm[:, 0].mean(dim=0)
    refc_spread = ensemble_denorm[:, 0].std(dim=0)
    cv = refc_spread / (torch.abs(refc_mean) + 1)  # Coefficient of variation
    
    im3 = ax3.imshow(cv, cmap='plasma', vmin=0, vmax=2, aspect='auto')
    ax3.set_title('Normalized Uncertainty\n(Spread/Mean)', fontsize=12)
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Coefficient of Variation')
    
    # 4. Temperature ensemble mean with contours
    ax4 = plt.subplot(3, 3, 4)
    t2m_mean = ensemble_denorm[:, 1].mean(dim=0)
    im4 = ax4.imshow(t2m_mean, cmap='RdBu_r', vmin=270, vmax=310, aspect='auto')
    
    # Add contours every 5K
    contour_levels = np.arange(270, 311, 5)
    cs = ax4.contour(t2m_mean, levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
    ax4.clabel(cs, inline=True, fontsize=8)
    
    ax4.set_title('2m Temperature Ensemble Mean (K)', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Wind speed and uncertainty
    ax5 = plt.subplot(3, 3, 5)
    u10 = ensemble_denorm[:, 3]
    v10 = ensemble_denorm[:, 4]
    wind_speed = torch.sqrt(u10**2 + v10**2)
    wind_mean = wind_speed.mean(dim=0)
    wind_spread = wind_speed.std(dim=0)
    
    im5 = ax5.imshow(wind_mean, cmap='viridis', vmin=0, vmax=20, aspect='auto')
    ax5.set_title('10m Wind Speed Mean (m/s)', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Spaghetti plot for specific contour
    ax6 = plt.subplot(3, 3, 6)
    # Show background mean
    im6 = ax6.imshow(refc_mean, cmap='gray', vmin=-20, vmax=60, alpha=0.3, aspect='auto')
    
    # Plot 35 dBZ contours from each member
    for m in range(0, num_members, 2):  # Every other member for clarity
        cs = ax6.contour(ensemble_denorm[m, 0], levels=[35], colors='red', 
                        alpha=0.5, linewidths=1)
    
    ax6.set_title('35 dBZ Reflectivity Contours\n(Ensemble Spaghetti)', fontsize=12)
    ax6.axis('off')
    
    # 7-9. Difference plots
    for i, (var_idx, var_name, cmap) in enumerate([(1, 'T2M Change', 'RdBu_r'),
                                                    (5, 'CAPE Change', 'BrBG'),
                                                    (0, 'REFC Change', 'RdBu_r')]):
        ax = plt.subplot(3, 3, 7 + i)
        diff = ensemble_denorm[:, var_idx].mean(dim=0) - current_denorm[0, var_idx]
        diff_limit = torch.abs(diff).max()
        
        im = ax.imshow(diff, cmap=cmap, vmin=-diff_limit, vmax=diff_limit, aspect='auto')
        ax.set_title(f'{var_name} (+1h)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'DDPM Ensemble Forecast Showcase - {num_members} Members\n'
                 f'Valid: {datetime.now() + timedelta(hours=1):%Y-%m-%d %H:00 UTC} (+1h)',
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save outputs
    output_dir = Path('forecasts')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'ensemble_showcase_{timestamp}.png'
    plt.savefig(plot_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved showcase to: {plot_file}")
    
    # Print interesting statistics
    print("\n" + "="*60)
    print("ENSEMBLE FORECAST STATISTICS")
    print("="*60)
    
    print("\nReflectivity Probabilities:")
    print(f"  P(REFC > 20 dBZ): {refc_prob_20.mean().item()*100:.1f}%")
    print(f"  P(REFC > 35 dBZ): {refc_prob_35.mean().item()*100:.1f}%")
    print(f"  P(REFC > 50 dBZ): {refc_prob_50.mean().item()*100:.1f}%")
    
    print("\nConvective Indicators:")
    print(f"  P(CAPE > 1000 J/kg): {cape_prob_1000.mean().item()*100:.1f}%")
    print(f"  P(CAPE > 2500 J/kg): {cape_prob_2500.mean().item()*100:.1f}%")
    print(f"  Max CAPE in ensemble: {ensemble_denorm[:, 5].max().item():.0f} J/kg")
    
    print("\nEnsemble Spread (domain average):")
    for i, var in enumerate(['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN']):
        spread = ensemble_denorm[:, i].std(dim=0).mean().item()
        print(f"  {var}: {spread:.2f}")
    
    print("\n✓ Showcase forecast complete!")


if __name__ == '__main__':
    generate_showcase_forecast()