#!/usr/bin/env python3
"""
Evaluation of trained DDPM model - generates ensemble forecasts and computes metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


class DDPMEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model
        self.model = UltraMinimalDDPM(
            in_channels=7,
            out_channels=7,
            base_dim=16
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load config
        self.config = checkpoint['config']
        
        # Create noise schedule
        self.noise_schedule = CosineBetaSchedule(
            timesteps=self.config['diffusion']['timesteps']
        )
        
        # Load normalizer
        self.normalizer = Normalizer(self.config['data']['stats'])
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['loss']:.4f}")
        
    def generate_ensemble(self, num_members=50, num_steps=50):
        """Generate ensemble forecast using DDPM sampling."""
        # Get sample shape from dataset
        C = 7  # Number of channels
        H, W = 1059, 1799  # Full HRRR resolution
        
        print(f"\nGenerating {num_members} ensemble members...")
        ensemble = []
        
        for i in tqdm(range(num_members), desc="Ensemble generation"):
            # Start from noise
            x = torch.randn(1, C, H, W, device=self.device)
            
            # Reverse diffusion process
            with torch.no_grad():
                for t in reversed(range(num_steps)):
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    # Predict noise
                    noise_pred = self.model(x, t_tensor)
                    
                    # DDPM sampling step
                    alpha = self.noise_schedule.alphas[t]
                    alpha_bar = self.noise_schedule.alphas_cumprod[t]
                    
                    if t > 0:
                        alpha_bar_prev = self.noise_schedule.alphas_cumprod[t-1]
                        beta = 1 - alpha
                        
                        # Compute x_{t-1} from x_t
                        mean = (x - beta / torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
                        
                        # Add noise for next step
                        var = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta
                        noise = torch.randn_like(x) * torch.sqrt(var)
                        x = mean + noise
                    else:
                        # Final step - no noise
                        x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
            
            ensemble.append(x.cpu())
        
        # Stack ensemble
        ensemble = torch.cat(ensemble, dim=0)
        return ensemble
    
    def compute_ensemble_metrics(self, ensemble):
        """Compute metrics for the generated ensemble."""
        # Ensemble statistics
        ens_mean = ensemble.mean(dim=0)
        ens_std = ensemble.std(dim=0)
        
        metrics = {
            'ensemble_size': ensemble.shape[0],
            'mean_spread': ens_std.mean().item(),
            'max_spread': ens_std.max().item(),
            'min_spread': ens_std.min().item(),
        }
        
        # Per-variable statistics
        var_names = self.config['data']['variables']
        for i, var in enumerate(var_names):
            metrics[f'{var}_mean_spread'] = ens_std[i].mean().item()
            metrics[f'{var}_max_spread'] = ens_std[i].max().item()
        
        return metrics, ens_mean, ens_std
    
    def plot_ensemble_results(self, ens_mean, ens_std, save_path):
        """Create visualization of ensemble mean and spread."""
        var_names = self.config['data']['variables']
        
        # Create figure with subplots for key variables
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Variables to plot (indices)
        plot_vars = [0, 1, 5]  # REFC, T2M, CAPE
        
        for idx, (var_idx, ax) in enumerate(zip(plot_vars, axes[:3])):
            # Plot ensemble mean
            var_name = var_names[var_idx]
            im = ax.imshow(ens_mean[var_idx].numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'{var_name} - Ensemble Mean')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for idx, (var_idx, ax) in enumerate(zip(plot_vars, axes[3:])):
            # Plot ensemble spread
            var_name = var_names[var_idx]
            im = ax.imshow(ens_std[var_idx].numpy(), cmap='plasma', aspect='auto')
            ax.set_title(f'{var_name} - Ensemble Spread')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {save_path}")
    
    def save_ensemble_netcdf(self, ensemble, save_path):
        """Save ensemble to NetCDF file."""
        var_names = self.config['data']['variables']
        
        # Create dataset
        data_vars = {}
        for i, var in enumerate(var_names):
            data_vars[var] = xr.DataArray(
                ensemble[:, i].numpy(),
                dims=['member', 'y', 'x'],
                attrs={'long_name': var}
            )
        
        # Add ensemble statistics
        ens_mean = ensemble.mean(dim=0)
        ens_std = ensemble.std(dim=0)
        
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
        
        ds = xr.Dataset(
            data_vars,
            coords={
                'member': np.arange(ensemble.shape[0]),
                'y': np.arange(ensemble.shape[2]),
                'x': np.arange(ensemble.shape[3])
            },
            attrs={
                'title': 'DDPM Ensemble Forecast',
                'model': 'Ultra-minimal DDPM (81K params)',
                'created': datetime.now().isoformat(),
                'training_epochs': 133,
                'final_loss': 0.0698
            }
        )
        
        ds.to_netcdf(save_path)
        print(f"Saved ensemble to {save_path}")


def main():
    # Setup
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create evaluator
    evaluator = DDPMEvaluator(checkpoint_path)
    
    # Generate ensemble
    ensemble = evaluator.generate_ensemble(num_members=50, num_steps=100)
    
    # Compute metrics
    metrics, ens_mean, ens_std = evaluator.compute_ensemble_metrics(ensemble)
    
    # Save metrics
    with open(output_dir / 'ensemble_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nEnsemble Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Create visualizations
    evaluator.plot_ensemble_results(
        ens_mean, ens_std,
        output_dir / 'ensemble_visualization.png'
    )
    
    # Save ensemble data
    evaluator.save_ensemble_netcdf(
        ensemble,
        output_dir / 'ensemble_forecast.nc'
    )
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()