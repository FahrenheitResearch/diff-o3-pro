#!/usr/bin/env python
"""
Ensemble inference for DEF: Generate probabilistic weather forecasts.
Implements Equations 20-21 from the paper for ensemble generation.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import zarr
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import json
from tqdm import tqdm
from typing import List, Optional, Tuple

from models.unet_attention_fixed import UNetAttn
from models.diffusion import GaussianDiffusion, ConditionalDiffusionUNet, DPMSolverPlusPlus
from utils.normalization import Normalizer
import utils.metrics as metrics


def load_config(path: str = "configs/expanded.yaml") -> dict:
    """Load configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_models(cfg: dict, device: torch.device) -> Tuple[nn.Module, nn.Module, GaussianDiffusion]:
    """Load trained forecast and diffusion models."""
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config
        total_channels = len(cfg['data']['variables'])
    else:
        # Full config
        n_surface = len(cfg['data']['surface_variables'])
        n_3d = len(cfg['data']['atmospheric_variables']) * len(cfg['data']['pressure_levels'])
        n_forcing = len(cfg['data'].get('forcing_variables', []))
        total_channels = n_surface + n_3d + n_forcing
    
    print(f"Model channels: {total_channels}")
    
    # Load forecast model
    forecast_model = UNetAttn(
        total_channels,
        total_channels,
        base_features=cfg['training']['base_features'],
        use_temporal_encoding=True
    )
    
    forecast_ckpt = torch.load('checkpoints/forecast_model_best.pt', map_location=device)
    forecast_model.load_state_dict(forecast_ckpt['model_state_dict'])
    forecast_model.to(device).eval()
    print("✓ Loaded forecast model")
    
    # Load diffusion model
    diffusion_process = GaussianDiffusion(
        timesteps=cfg['diffusion']['timesteps'],
        beta_schedule=cfg['diffusion']['beta_schedule']
    ).to(device)
    
    diffusion_model = ConditionalDiffusionUNet(
        in_channels=total_channels,
        out_channels=total_channels,
        cond_channels=total_channels,
        base_features=cfg['training']['base_features']
    )
    
    diffusion_ckpt = torch.load('checkpoints/diffusion/best_model.pt', map_location=device)
    diffusion_model.load_state_dict(diffusion_ckpt['model_state_dict'])
    diffusion_model.to(device).eval()
    print("✓ Loaded diffusion model")
    
    return forecast_model, diffusion_model, diffusion_process


def generate_ensemble_perturbations(
    x0: torch.Tensor,
    forecast_model: nn.Module,
    diffusion_model: nn.Module,
    diffusion_process: GaussianDiffusion,
    cfg: dict,
    device: torch.device,
    timestamps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generate ensemble of perturbed initial conditions using diffusion model.
    Implements Equations 20-21 from the DEF paper.
    
    Args:
        x0: Initial state [1, C, H, W]
        forecast_model: Deterministic forecast model G_φ
        diffusion_model: Diffusion model ε_θ
        diffusion_process: Diffusion process handler
        cfg: Configuration
        device: Device
        timestamps: Time information for temporal encoding
        
    Returns:
        Ensemble of perturbed states [B, C, H, W]
    """
    B = cfg['ensemble']['num_members']  # Number of ensemble members
    K = cfg['ensemble']['perturbation_samples']  # Number of perturbation samples
    omega = cfg['ensemble']['blend_weight']  # Blending weight
    
    print(f"Ensemble config: B={B}, K={K}, omega={omega}")
    
    # Get deterministic forecast (condition for diffusion)
    print("Getting deterministic forecast...")
    with torch.no_grad():
        x0_forecast = forecast_model(x0, timestamps)
    print(f"Forecast shape: {x0_forecast.shape}")
    
    # Initialize DPM-Solver++
    dpm_solver = DPMSolverPlusPlus(
        noise_prediction_fn=lambda x, t, c: diffusion_model(x, t, c),
        alphas_cumprod=diffusion_process.alphas_cumprod,
        num_timesteps=cfg['diffusion']['timesteps'],
        solver_order=cfg['diffusion']['solver_order']
    )
    
    # Generate ensemble members
    ensemble = []
    
    print(f"Generating {B} ensemble members...")
    
    for b in range(B):
        print(f"  Member {b+1}/{B}")
        # Sample K perturbations from diffusion model
        perturbations = []
        
        for k in range(K):
            print(f"    Perturbation {k+1}/{K}")
            # Start from noise
            z_T = torch.randn_like(x0)
            
            # Generate sample using DPM-Solver++
            print(f"    Sampling with {cfg['diffusion']['num_steps']} steps...")
            x_perturbed = dpm_solver.sample(
                z_T,
                num_steps=cfg['diffusion']['num_steps'],
                condition=x0_forecast,
                guidance_weight=cfg['diffusion']['guidance_weight']
            )
            print(f"    Done sampling")
            
            perturbations.append(x_perturbed)
        
        # Average perturbations (Eq. 21)
        x_tilde = torch.stack(perturbations, dim=0).mean(dim=0)
        
        # Blend with original state (Eq. 20)
        x_blend = omega * x0 + (1 - omega) * x_tilde
        
        ensemble.append(x_blend)
    
    # Stack ensemble members
    ensemble = torch.cat(ensemble, dim=0)  # [B, C, H, W]
    
    return ensemble


def autoregressive_forecast(
    initial_state: torch.Tensor,
    forecast_model: nn.Module,
    lead_hours: int,
    base_timestamp: float,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Generate autoregressive forecast for specified lead time.
    
    Args:
        initial_state: Starting state [B, C, H, W]
        forecast_model: Deterministic forecast model
        lead_hours: Number of hours to forecast
        base_timestamp: Starting timestamp (hours since epoch)
        device: Device
        
    Returns:
        List of states at each hour
    """
    states = [initial_state]
    current_state = initial_state
    
    for hour in range(lead_hours):
        # Create timestamps for temporal encoding
        timestamps = torch.full(
            (current_state.shape[0],),
            base_timestamp + hour,
            device=device
        )
        
        # Forecast next hour
        with torch.no_grad():
            next_state = forecast_model(current_state, timestamps)
        
        states.append(next_state)
        current_state = next_state
    
    return states


def process_forecast_cycle(
    initial_data: dict,
    forecast_model: nn.Module,
    diffusion_model: nn.Module,
    diffusion_process: GaussianDiffusion,
    cfg: dict,
    device: torch.device,
    cycle_time: datetime,
    max_lead_hours: int = 240
) -> xr.Dataset:
    """
    Process a complete forecast cycle with ensemble generation.
    
    Args:
        initial_data: Dictionary with initial state data
        forecast_model: Deterministic forecast model
        diffusion_model: Diffusion model
        diffusion_process: Diffusion process
        cfg: Configuration
        device: Device
        cycle_time: Forecast initialization time
        max_lead_hours: Maximum forecast lead time
        
    Returns:
        xarray Dataset with ensemble forecasts
    """
    # Convert initial data to tensor
    x0 = torch.tensor(initial_data['data'], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate timestamp
    base_timestamp = (cycle_time - datetime(2020, 1, 1)).total_seconds() / 3600.0
    timestamps = torch.tensor([base_timestamp], device=device)
    
    print(f"\nGenerating ensemble for cycle {cycle_time}...")
    print(f"Initial state shape: {x0.shape}")
    print(f"Device: {device}")
    
    # Generate perturbed ensemble members
    print("Calling generate_ensemble_perturbations...")
    ensemble_init = generate_ensemble_perturbations(
        x0, forecast_model, diffusion_model, diffusion_process,
        cfg, device, timestamps
    )
    
    print(f"Generated {ensemble_init.shape[0]} ensemble members")
    
    # Run autoregressive forecast for each ensemble member
    ensemble_forecasts = []
    
    for i in tqdm(range(ensemble_init.shape[0]), desc="Ensemble members"):
        member_init = ensemble_init[i:i+1]  # Keep batch dimension
        
        # Generate forecast trajectory
        trajectory = autoregressive_forecast(
            member_init,
            forecast_model,
            max_lead_hours,
            base_timestamp,
            device
        )
        
        # Stack trajectory [T+1, C, H, W]
        trajectory = torch.cat(trajectory, dim=0)
        ensemble_forecasts.append(trajectory.cpu())
    
    # Stack ensemble [B, T+1, C, H, W]
    ensemble_forecasts = torch.stack(ensemble_forecasts, dim=0)
    
    # Create xarray dataset
    lead_times = np.arange(0, max_lead_hours + 1)
    valid_times = [cycle_time + timedelta(hours=int(h)) for h in lead_times]
    
    # Get variable names
    var_names = initial_data['variables']
    
    # Create data arrays
    data_vars = {}
    
    # Split ensemble data by variable
    for i, var in enumerate(var_names):
        data_vars[var] = xr.DataArray(
            ensemble_forecasts[:, :, i].numpy(),
            dims=['member', 'time', 'y', 'x'],
            coords={
                'member': np.arange(ensemble_forecasts.shape[0]),
                'time': valid_times,
                'init_time': cycle_time
            },
            attrs={
                'long_name': var,
                'units': initial_data.get('units', {}).get(var, ''),
                'init_time': cycle_time.isoformat()
            }
        )
    
    # Add ensemble mean
    ensemble_mean = ensemble_forecasts.mean(dim=0)
    for i, var in enumerate(var_names):
        data_vars[f"{var}_mean"] = xr.DataArray(
            ensemble_mean[:, i].numpy(),
            dims=['time', 'y', 'x'],
            coords={
                'time': valid_times,
                'init_time': cycle_time
            }
        )
    
    # Add ensemble spread
    ensemble_spread = ensemble_forecasts.std(dim=0)
    for i, var in enumerate(var_names):
        data_vars[f"{var}_spread"] = xr.DataArray(
            ensemble_spread[:, i].numpy(),
            dims=['time', 'y', 'x'],
            coords={
                'time': valid_times,
                'init_time': cycle_time
            }
        )
    
    # Create dataset
    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': 'DEF Ensemble Weather Forecast',
            'institution': 'HRRR-DEF',
            'source': 'Diffusion Ensemble Forecasting',
            'init_time': cycle_time.isoformat(),
            'ensemble_size': ensemble_forecasts.shape[0],
            'max_lead_hours': max_lead_hours,
            'created': datetime.now().isoformat()
        }
    )
    
    # Add coordinates
    if 'latitude' in initial_data:
        ds.coords['latitude'] = (('y', 'x'), initial_data['latitude'])
    if 'longitude' in initial_data:
        ds.coords['longitude'] = (('y', 'x'), initial_data['longitude'])
    
    return ds


def load_initial_conditions(
    zarr_path: Path,
    variable_list: List[str],
    timestamp: datetime,
    normalizer: Normalizer
) -> dict:
    """Load and prepare initial conditions from Zarr archive."""
    store = zarr.open(str(zarr_path), mode='r')
    
    # Find closest time index
    times = store['time'][:]
    # Convert timestamp to hours (assuming times are hours from some epoch)
    # For simplicity, just use the first available time
    time_idx = 0
    print(f"Using time index {time_idx} (time={times[time_idx]})")
    
    # Load data for all variables
    data = []
    for var in variable_list:
        if var in store:
            var_data = store[var][time_idx].astype(np.float32)
            # Normalize
            var_data = normalizer.encode(var_data, var)
            data.append(var_data)
        else:
            print(f"Warning: Variable {var} not found in Zarr store")
            # Add zeros as placeholder
            shape = store[variable_list[0]][time_idx].shape
            data.append(np.zeros(shape, dtype=np.float32))
    
    # Stack variables
    data = np.stack(data, axis=0)  # [C, H, W]
    
    # Get coordinates if available
    result = {
        'data': data,
        'variables': variable_list,
        'timestamp': timestamp
    }
    
    if 'latitude' in store:
        result['latitude'] = store['latitude'][:]
    if 'longitude' in store:
        result['longitude'] = store['longitude'][:]
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate ensemble weather forecasts with DEF')
    parser.add_argument('--config', type=str, default='configs/expanded.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date for forecasts (YYYY-MM-DD HH)')
    parser.add_argument('--cycles', type=int, default=1,
                        help='Number of forecast cycles to process')
    parser.add_argument('--cycle-interval', type=int, default=6,
                        help='Hours between forecast cycles')
    parser.add_argument('--max-lead-hours', type=int, default=240,
                        help='Maximum forecast lead time in hours')
    parser.add_argument('--output-dir', type=Path, default=Path('forecasts'),
                        help='Output directory for forecasts')
    parser.add_argument('--zarr-path', type=Path, default=None,
                        help='Path to Zarr data (overrides config)')
    parser.add_argument('--ensemble-size', type=int, default=None,
                        help='Number of ensemble members (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override config with command line arguments
    if args.zarr_path:
        cfg['data']['zarr'] = str(args.zarr_path)
    if args.ensemble_size:
        cfg['ensemble']['num_members'] = args.ensemble_size
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand variable list - handle both test and full config
    if 'variables' in cfg['data']:
        # Simple test config
        all_vars = cfg['data']['variables']
    else:
        # Full config
        all_vars = cfg['data']['surface_variables'].copy()
        for var in cfg['data']['atmospheric_variables']:
            for level in cfg['data']['pressure_levels']:
                all_vars.append(f"{var}_{level}")
        all_vars.extend(cfg['data'].get('forcing_variables', []))
    
    print(f"Total variables: {len(all_vars)}")
    
    # Load models
    forecast_model, diffusion_model, diffusion_process = load_models(cfg, device)
    
    # Initialize normalizer
    normalizer = Normalizer(Path(cfg['data']['stats']))
    
    # Parse start date
    start_time = datetime.strptime(args.start_date, '%Y-%m-%d %H')
    
    # Process each forecast cycle
    for cycle_idx in range(args.cycles):
        cycle_time = start_time + timedelta(hours=cycle_idx * args.cycle_interval)
        print(f"\n{'='*60}")
        print(f"Processing forecast cycle: {cycle_time}")
        print(f"{'='*60}")
        
        # Load initial conditions
        initial_data = load_initial_conditions(
            Path(cfg['data']['zarr']),
            all_vars,
            cycle_time,
            normalizer
        )
        
        # Generate ensemble forecast
        forecast_ds = process_forecast_cycle(
            initial_data,
            forecast_model,
            diffusion_model,
            diffusion_process,
            cfg,
            device,
            cycle_time,
            args.max_lead_hours
        )
        
        # Save forecast
        output_file = args.output_dir / f"def_forecast_{cycle_time.strftime('%Y%m%d_%H')}Z.nc"
        forecast_ds.to_netcdf(output_file)
        print(f"\nSaved forecast to: {output_file}")
        
        # Compute and print some basic statistics
        print("\nForecast statistics:")
        for lead_hour in [6, 12, 24, 48, 72, 120, 240]:
            if lead_hour <= args.max_lead_hours:
                time_idx = lead_hour
                t850_mean = forecast_ds[f"T_850_mean"].isel(time=time_idx).mean().item()
                t850_spread = forecast_ds[f"T_850_spread"].isel(time=time_idx).mean().item()
                print(f"  T@850hPa +{lead_hour:3d}h: mean={t850_mean:.2f}, spread={t850_spread:.2f}")
    
    print("\n✓ Ensemble forecast generation complete!")


if __name__ == '__main__':
    main()