#!/usr/bin/env python3
"""
Generate demonstration ensemble forecast with visualizations.
Optimized for speed with fewer ensemble members.
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def generate_demo_forecast():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = UltraMinimalDDPM(
        in_channels=7,
        out_channels=7,
        base_dim=16
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    config = checkpoint['config']
    print(f"Model loaded - Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
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
    print("\nLoading HRRR data...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Get a sample with interesting weather
    print("Finding interesting weather case...")
    best_idx = 0
    max_cape = 0
    
    # Look for high CAPE case (convective potential)
    for i in range(min(50, len(dataset))):
        current, future = dataset[i]
        cape_value = current[5].max().item()  # CAPE is index 5
        if cape_value > max_cape:
            max_cape = cape_value
            best_idx = i
    
    print(f"Selected case with max CAPE: {max_cape:.1f}")
    current_state, future_state = dataset[best_idx]
    current_state = current_state.unsqueeze(0).to(device)
    
    # Generate smaller ensemble for speed
    num_members = 10
    num_steps = 50  # Fewer diffusion steps
    
    print(f"\nGenerating {num_members}-member ensemble forecast...")
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
    print("\nDenormalizing forecast...")
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
    
    # Compute statistics
    ens_mean = ensemble_denorm.mean(dim=0)
    ens_std = ensemble_denorm.std(dim=0)
    
    # Create comprehensive visualization
    print("\nCreating visualizations...")
    fig = plt.figure(figsize=(20, 24))
    
    # Variables to plot with better colormaps
    plot_configs = [
        ('REFC', 0, 'Composite Reflectivity (dBZ)', 'turbo', 0, 70),
        ('T2M', 1, '2m Temperature (K)', 'RdBu_r', 250, 320),
        ('CAPE', 5, 'CAPE (J/kg)', 'hot_r', 0, 4000),
        ('CIN', 6, 'CIN (J/kg)', 'Blues_r', -300, 0)
    ]
    
    # Create subplots
    for i, (var_name, idx, title, cmap, vmin, vmax) in enumerate(plot_configs):
        # Current state
        ax1 = plt.subplot(4, 4, i*4 + 1)
        im1 = ax1.imshow(current_denorm[0, idx], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax1.set_title(f'{title}\nCurrent State')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Ensemble mean
        ax2 = plt.subplot(4, 4, i*4 + 2)
        im2 = ax2.imshow(ens_mean[idx], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title(f'{title}\nEnsemble Mean (+1h)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Ensemble spread
        ax3 = plt.subplot(4, 4, i*4 + 3)
        im3 = ax3.imshow(ens_std[idx], cmap='YlOrRd', aspect='auto')
        ax3.set_title(f'{title}\nEnsemble Spread')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Difference (mean - current)
        ax4 = plt.subplot(4, 4, i*4 + 4)
        diff = ens_mean[idx] - current_denorm[0, idx]
        diff_limit = torch.abs(diff).max()
        im4 = ax4.imshow(diff, cmap='RdBu_r', vmin=-diff_limit, vmax=diff_limit, aspect='auto')
        ax4.set_title(f'{title}\nChange (+1h)')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    plt.suptitle(f'DDPM Ensemble Forecast - {num_members} Members\n'
                 f'Valid Time: {datetime.now() + timedelta(hours=1):%Y-%m-%d %H:00 UTC} (+1h)',
                 fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('forecasts')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'ensemble_forecast_demo_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {plot_file}")
    
    # Print statistics
    print("\nForecast Statistics:")
    print("-" * 50)
    for var_name, idx, _, _, _, _ in plot_configs:
        mean_val = ens_mean[idx].mean().item()
        spread_val = ens_std[idx].mean().item()
        max_spread = ens_std[idx].max().item()
        print(f"{var_name:6s}: mean={mean_val:8.2f}, avg_spread={spread_val:6.2f}, max_spread={max_spread:6.2f}")
    
    # Save ensemble data
    print("\nSaving ensemble data...")
    data_vars = {}
    for i, var in enumerate(var_names):
        data_vars[f'{var}_ensemble'] = xr.DataArray(
            ensemble_denorm[:, i].numpy(),
            dims=['member', 'y', 'x']
        )
        data_vars[f'{var}_mean'] = xr.DataArray(
            ens_mean[i].numpy(),
            dims=['y', 'x']
        )
        data_vars[f'{var}_spread'] = xr.DataArray(
            ens_std[i].numpy(),
            dims=['y', 'x']
        )
    
    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': 'DDPM Ensemble Forecast Demo',
            'model': 'Ultra-minimal DDPM (81K params)',
            'ensemble_size': num_members,
            'created': datetime.now().isoformat()
        }
    )
    
    nc_file = output_dir / f'ensemble_forecast_demo_{timestamp}.nc'
    ds.to_netcdf(nc_file)
    print(f"Saved NetCDF to: {nc_file}")
    
    print("\n✓ Forecast generation complete!")


if __name__ == '__main__':
    generate_demo_forecast()