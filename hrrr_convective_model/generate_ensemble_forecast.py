#!/usr/bin/env python3
"""
Generate ensemble forecasts using trained diffusion model.
This demonstrates the key innovation of DEF - calibrated uncertainty for convection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm

from models.diffusion.ddpm_convection import ConvectionDDPM, CosineBetaSchedule
from utils.normalization import Normalizer


def generate_ensemble_forecast(
    model_path: str,
    config_path: str,
    past_data: torch.Tensor,
    num_members: int = 50,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate ensemble forecast from trained diffusion model.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model config
        past_data: (T, C, H, W) past weather states
        num_members: Number of ensemble members to generate
        
    Returns:
        ensemble: (M, C, H, W) ensemble forecast
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(device)
    
    # Create model
    model = ConvectionDDPM(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=config['model']['base_dim'],
        dim_mults=tuple(config['model']['dim_mults']),
        attention_resolutions=tuple(config['model']['attention_resolutions']),
        num_res_blocks=config['model']['num_res_blocks'],
        dropout=0.0,  # No dropout at inference
        history_length=config['model']['history_length']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps']
    )
    
    # Move schedule to device
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 
                 'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Prepare past data
    past_data = past_data.unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate ensemble
    ensemble_members = []
    
    print(f"Generating {num_members} ensemble members...")
    for i in tqdm(range(num_members)):
        with torch.no_grad():
            # Start from pure noise
            shape = (1, len(config['data']['variables']), 1059, 1799)
            xt = torch.randn(shape, device=device)
            
            # Reverse diffusion process
            for t in reversed(range(config['diffusion']['timesteps'])):
                t_batch = torch.tensor([t], device=device)
                
                # Predict noise
                noise_pred = model(xt, t_batch, past_data)
                
                # Denoise step
                xt = noise_schedule.denoise_step(xt, noise_pred, t)
            
            ensemble_members.append(xt.squeeze(0).cpu())
    
    # Stack ensemble
    ensemble = torch.stack(ensemble_members, dim=0)  # (M, C, H, W)
    
    return ensemble


def plot_ensemble_forecast(ensemble, variable_idx=0, variable_name='REFC'):
    """
    Plot ensemble forecast with uncertainty visualization.
    Shows why DEF is critical for convection prediction.
    """
    # Compute statistics
    mean = ensemble[:, variable_idx].mean(dim=0)
    std = ensemble[:, variable_idx].std(dim=0)
    p10 = torch.quantile(ensemble[:, variable_idx], 0.1, dim=0)
    p90 = torch.quantile(ensemble[:, variable_idx], 0.9, dim=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean forecast
    im1 = axes[0, 0].imshow(mean, cmap='viridis')
    axes[0, 0].set_title(f'Ensemble Mean {variable_name}')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Standard deviation (uncertainty)
    im2 = axes[0, 1].imshow(std, cmap='hot')
    axes[0, 1].set_title('Ensemble Spread (Uncertainty)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 10th percentile
    im3 = axes[1, 0].imshow(p10, cmap='viridis')
    axes[1, 0].set_title('10th Percentile (Low scenario)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 90th percentile  
    im4 = axes[1, 1].imshow(p90, cmap='viridis')
    axes[1, 1].set_title('90th Percentile (High scenario)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.suptitle('DEF Ensemble Forecast - Calibrated Uncertainty for Convection')
    plt.tight_layout()
    
    return fig


def compute_convective_probabilities(ensemble, cape_idx=5, cin_idx=6, refc_idx=0):
    """
    Compute probabilistic convective forecasts from ensemble.
    This is what saves lives - knowing probability of severe weather.
    """
    # Thresholds for convection
    cape_threshold = 1000  # J/kg
    cin_threshold = -50   # J/kg
    refc_threshold = 35   # dBZ
    
    # Compute probabilities
    prob_cape = (ensemble[:, cape_idx] > cape_threshold).float().mean(dim=0)
    prob_cin = (ensemble[:, cin_idx] > cin_threshold).float().mean(dim=0)
    prob_refc = (ensemble[:, refc_idx] > refc_threshold).float().mean(dim=0)
    
    # Combined convective probability
    prob_convection = prob_cape * prob_cin * prob_refc
    
    return {
        'prob_cape': prob_cape,
        'prob_cin': prob_cin,
        'prob_refc': prob_refc,
        'prob_convection': prob_convection
    }


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--data', type=str, required=True, help='Input data path')
    parser.add_argument('--output', type=str, default='ensemble_forecast.nc')
    parser.add_argument('--members', type=int, default=50)
    args = parser.parse_args()
    
    # Load past data (implement based on your data format)
    # past_data = load_past_states(args.data)
    
    # Generate ensemble
    ensemble = generate_ensemble_forecast(
        args.model,
        args.config,
        past_data,
        num_members=args.members
    )
    
    # Compute convective probabilities
    conv_probs = compute_convective_probabilities(ensemble)
    
    # Save results
    # Save as NetCDF for further analysis
    
    print(f"Generated {args.members}-member ensemble forecast")
    print("Convection probability statistics:")
    print(f"  Max probability: {conv_probs['prob_convection'].max():.2%}")
    print(f"  Area > 50%: {(conv_probs['prob_convection'] > 0.5).sum().item()} grid points")