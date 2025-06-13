#!/usr/bin/env python3
"""
Generate ensemble forecast using trained DDPM model.
This creates a realistic weather forecast with uncertainty quantification.
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


class EnsembleForecastGenerator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model
        self.model = UltraMinimalDDPM(
            in_channels=7,
            out_channels=7,
            base_dim=16
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.config = checkpoint['config']
        print(f"Model loaded - Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        
        # Create noise schedule
        self.noise_schedule = CosineBetaSchedule(
            timesteps=self.config['diffusion']['timesteps'],
            s=self.config['diffusion']['s']
        )
        
        # Move schedule to device
        for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
            setattr(self.noise_schedule, attr, getattr(self.noise_schedule, attr).to(self.device))
        
        # Load normalizer
        self.normalizer = Normalizer(self.config['data']['stats'])
        
    def generate_ensemble_forecast(self, initial_state, num_members=50, num_diffusion_steps=100):
        """
        Generate ensemble forecast from initial atmospheric state.
        
        Args:
            initial_state: Current weather state [B, C, H, W] (normalized)
            num_members: Number of ensemble members to generate
            num_diffusion_steps: Number of denoising steps
            
        Returns:
            ensemble: Generated ensemble [num_members, C, H, W]
        """
        B, C, H, W = initial_state.shape
        
        print(f"\nGenerating {num_members}-member ensemble forecast...")
        ensemble_members = []
        
        # Generate each ensemble member
        for i in tqdm(range(num_members), desc="Ensemble generation"):
            # Start from Gaussian noise
            x = torch.randn(B, C, H, W, device=self.device)
            
            # Reverse diffusion process
            with torch.no_grad():
                # Use evenly spaced timesteps for efficiency
                timesteps = np.linspace(self.config['diffusion']['timesteps']-1, 0, 
                                      num_diffusion_steps, dtype=int)
                
                for t_idx in timesteps:
                    t = torch.tensor([t_idx], device=self.device)
                    
                    # Predict noise
                    noise_pred = self.model(x, t)
                    
                    # DDPM reverse step
                    alpha = self.noise_schedule.alphas[t_idx]
                    alpha_bar = self.noise_schedule.alphas_cumprod[t_idx]
                    beta = self.noise_schedule.betas[t_idx]
                    
                    if t_idx > 0:
                        alpha_bar_prev = self.noise_schedule.alphas_cumprod[t_idx - 1]
                        
                        # Predict x0
                        x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                        x0_pred = torch.clamp(x0_pred, -3, 3)
                        
                        # Compute posterior
                        posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                       (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                        
                        posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                        
                        # Add noise (except for last step)
                        noise = torch.randn_like(x) if t_idx > 1 else 0
                        x = posterior_mean + posterior_var.sqrt() * noise
                    else:
                        # Final denoising step
                        x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                        x = torch.clamp(x, -3, 3)
            
            # Add to ensemble
            ensemble_members.append(x.cpu())
        
        # Stack ensemble
        ensemble = torch.cat(ensemble_members, dim=0)
        return ensemble
    
    def create_forecast_dataset(self, ensemble, valid_time, lead_hour=1):
        """
        Create xarray dataset from ensemble forecast.
        
        Args:
            ensemble: Ensemble forecast tensor [num_members, C, H, W] (normalized)
            valid_time: Forecast valid time
            lead_hour: Lead time in hours
            
        Returns:
            xr.Dataset with ensemble forecast
        """
        var_names = self.config['data']['variables']
        num_members = ensemble.shape[0]
        
        # Denormalize ensemble
        ensemble_denorm = torch.zeros_like(ensemble)
        for i, var in enumerate(var_names):
            for m in range(num_members):
                ensemble_denorm[m, i] = torch.tensor(
                    self.normalizer.denormalize(ensemble[m, i].numpy(), var)
                )
        
        # Create data variables
        data_vars = {}
        
        # Individual ensemble members
        for i, var in enumerate(var_names):
            data_vars[var] = xr.DataArray(
                ensemble_denorm[:, i].numpy(),
                dims=['member', 'y', 'x'],
                attrs={
                    'long_name': var,
                    'units': self._get_units(var)
                }
            )
        
        # Ensemble statistics
        ens_mean = ensemble_denorm.mean(dim=0)
        ens_std = ensemble_denorm.std(dim=0)
        ens_min = ensemble_denorm.min(dim=0)[0]
        ens_max = ensemble_denorm.max(dim=0)[0]
        
        for i, var in enumerate(var_names):
            data_vars[f'{var}_mean'] = xr.DataArray(
                ens_mean[i].numpy(),
                dims=['y', 'x'],
                attrs={'long_name': f'{var} ensemble mean'}
            )
            data_vars[f'{var}_spread'] = xr.DataArray(
                ens_std[i].numpy(),
                dims=['y', 'x'],
                attrs={'long_name': f'{var} ensemble spread'}
            )
            data_vars[f'{var}_min'] = xr.DataArray(
                ens_min[i].numpy(),
                dims=['y', 'x'],
                attrs={'long_name': f'{var} ensemble minimum'}
            )
            data_vars[f'{var}_max'] = xr.DataArray(
                ens_max[i].numpy(),
                dims=['y', 'x'],
                attrs={'long_name': f'{var} ensemble maximum'}
            )
        
        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'member': np.arange(num_members),
                'y': np.arange(ensemble.shape[2]),
                'x': np.arange(ensemble.shape[3]),
                'valid_time': valid_time,
                'lead_hour': lead_hour
            },
            attrs={
                'title': 'DDPM Ensemble Weather Forecast',
                'institution': 'HRRR-DEF',
                'model': 'Ultra-minimal DDPM (81K params)',
                'created': datetime.now().isoformat(),
                'ensemble_size': num_members,
                'lead_time_hours': lead_hour
            }
        )
        
        return ds
    
    def _get_units(self, var):
        """Get units for each variable."""
        units_map = {
            'REFC': 'dBZ',
            'T2M': 'K',
            'D2M': 'K',
            'U10': 'm/s',
            'V10': 'm/s',
            'CAPE': 'J/kg',
            'CIN': 'J/kg'
        }
        return units_map.get(var, '')
    
    def plot_forecast(self, ds, save_path):
        """Create forecast visualization."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        axes = axes.flatten()
        
        # Variables to plot
        plot_vars = [
            ('REFC_mean', 'REFC_spread', 'Composite Reflectivity', 'dBZ', 'viridis', 'plasma'),
            ('T2M_mean', 'T2M_spread', '2m Temperature', 'K', 'RdBu_r', 'YlOrRd'),
            ('CAPE_mean', 'CAPE_spread', 'CAPE', 'J/kg', 'hot_r', 'YlOrRd')
        ]
        
        for i, (mean_var, spread_var, title, units, cmap_mean, cmap_spread) in enumerate(plot_vars):
            # Plot mean
            im = axes[i*2].imshow(ds[mean_var].values, cmap=cmap_mean, aspect='auto')
            axes[i*2].set_title(f'{title} - Ensemble Mean')
            axes[i*2].axis('off')
            cbar = plt.colorbar(im, ax=axes[i*2], fraction=0.046)
            cbar.set_label(units)
            
            # Plot spread
            im = axes[i*2+1].imshow(ds[spread_var].values, cmap=cmap_spread, aspect='auto')
            axes[i*2+1].set_title(f'{title} - Ensemble Spread')
            axes[i*2+1].axis('off')
            cbar = plt.colorbar(im, ax=axes[i*2+1], fraction=0.046)
            cbar.set_label(units)
        
        plt.suptitle(f"Ensemble Forecast - Valid: {ds.valid_time.values} (+{ds.lead_hour.values}h)", 
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved forecast plot to: {save_path}")


def main():
    # Configuration
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    output_dir = Path('forecasts')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = EnsembleForecastGenerator(checkpoint_path)
    
    # Load a sample from the dataset
    print("\nLoading sample data...")
    dataset = HRRRDataset(
        zarr_path=generator.config['data']['zarr'],
        variables=generator.config['data']['variables'],
        lead_hours=1,
        stats_path=generator.config['data']['stats']
    )
    
    # Get a random sample
    idx = np.random.randint(len(dataset))
    current_state, future_state = dataset[idx]
    current_state = current_state.unsqueeze(0).to(generator.device)
    
    # Generate ensemble forecast
    ensemble = generator.generate_ensemble_forecast(
        current_state,
        num_members=50,
        num_diffusion_steps=100
    )
    
    # Create forecast dataset
    valid_time = datetime.now() + timedelta(hours=1)
    forecast_ds = generator.create_forecast_dataset(
        ensemble,
        valid_time=valid_time,
        lead_hour=1
    )
    
    # Save forecast
    output_file = output_dir / f"ensemble_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
    forecast_ds.to_netcdf(output_file)
    print(f"\nSaved forecast to: {output_file}")
    
    # Create visualization
    plot_file = output_dir / f"ensemble_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    generator.plot_forecast(forecast_ds, plot_file)
    
    # Print statistics
    print("\nForecast Statistics:")
    for var in generator.config['data']['variables']:
        mean_spread = forecast_ds[f'{var}_spread'].mean().item()
        max_spread = forecast_ds[f'{var}_spread'].max().item()
        print(f"  {var}: mean_spread={mean_spread:.3f}, max_spread={max_spread:.3f}")
    
    print("\n✓ Ensemble forecast generation complete!")


if __name__ == '__main__':
    main()