#!/usr/bin/env python3
"""
Deterministic iterative forecasting with the fixed model.
Generates multi-hour forecasts by iterating 1-hour predictions.
"""

import torch
import numpy as np
import xarray as xr
import zarr
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from models.unet_residual import UNetResidual
from utils.normalization import Normalizer

def load_model(checkpoint_path, device):
    """Load trained deterministic model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = UNetResidual(
        in_ch=len(config['data']['variables']),
        out_ch=len(config['data']['variables']),
        base_features=config['training']['base_features'],
        use_temporal_encoding=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, config

def iterative_forecast(model, initial_state, initial_timestamp, num_hours, device):
    """Generate iterative forecast by chaining 1-hour predictions."""
    states = [initial_state]
    current_state = initial_state
    current_time = initial_timestamp
    
    print(f"Generating {num_hours}-hour forecast iteratively...")
    
    with torch.no_grad():
        for hour in range(num_hours):
            # Predict next hour
            next_state = model(current_state, current_time)
            states.append(next_state)
            
            # Update for next iteration
            current_state = next_state
            current_time = current_time + 1.0  # Add 1 hour
            
            if (hour + 1) % 6 == 0:
                print(f"  Completed {hour + 1} hours")
    
    # Stack all states [T+1, B, C, H, W]
    return torch.stack(states, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/deterministic/best_model.pt',
                       help='Model checkpoint path')
    parser.add_argument('--hours', type=int, default=24,
                       help='Number of hours to forecast')
    parser.add_argument('--output', type=str, default='forecast_deterministic.nc',
                       help='Output netCDF file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    variables = config['data']['variables']
    
    # Load normalizer
    normalizer = Normalizer(Path(config['data']['stats']))
    
    # Load initial conditions from latest data
    print("Loading initial conditions...")
    store = zarr.open('data/zarr/latest/hrrr.zarr', 'r')
    
    # Get latest available time
    initial_data = []
    for var in variables:
        data = store[var][0].astype(np.float32)
        # Normalize
        data_norm = normalizer.encode(data, var)
        initial_data.append(data_norm)
    
    # Convert to tensor
    x0 = torch.tensor(np.stack(initial_data, axis=0)).unsqueeze(0).to(device)
    
    # Set initial timestamp (hours since epoch)
    # This should come from the data in production
    initial_timestamp = torch.tensor([config['data'].get('epoch_start_hours', 0)], 
                                   device=device)
    
    # Generate forecast
    forecast_states = iterative_forecast(model, x0, initial_timestamp, 
                                       args.hours, device)
    
    # Convert back to numpy and denormalize
    print("Denormalizing forecast...")
    forecast_np = forecast_states.cpu().numpy()[:, 0]  # Remove batch dim
    
    forecast_denorm = np.zeros_like(forecast_np)
    for t in range(forecast_np.shape[0]):
        for v, var in enumerate(variables):
            forecast_denorm[t, v] = normalizer.decode(forecast_np[t, v], var)
    
    # Create xarray dataset
    print("Creating output dataset...")
    init_time = datetime.utcnow()
    valid_times = [init_time + timedelta(hours=h) for h in range(args.hours + 1)]
    
    data_vars = {}
    for v, var in enumerate(variables):
        data_vars[var] = xr.DataArray(
            forecast_denorm[:, v],
            dims=['time', 'y', 'x'],
            coords={'time': valid_times}
        )
    
    ds = xr.Dataset(data_vars, attrs={
        'title': 'Deterministic Weather Forecast',
        'model': 'UNetResidual',
        'init_time': init_time.isoformat(),
        'forecast_hours': args.hours
    })
    
    # Save forecast
    ds.to_netcdf(args.output)
    print(f"Saved forecast to: {args.output}")
    
    # Generate plots if requested
    if args.plot:
        print("Generating plots...")
        
        # Plot reflectivity evolution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        times_to_plot = [0, 1, 3, 6, 12, 24]
        times_to_plot = [t for t in times_to_plot if t <= args.hours]
        
        for i, t in enumerate(times_to_plot[:6]):
            ax = axes[i]
            
            if 'REFC' in variables:
                refc_idx = variables.index('REFC')
                data = forecast_denorm[t, refc_idx]
                
                # Mask low values
                data = np.where(data < -10, np.nan, data)
                
                im = ax.imshow(data, origin='lower', cmap='turbo', 
                             vmin=-10, vmax=60, aspect='auto')
                ax.set_title(f'+{t}h')
                plt.colorbar(im, ax=ax, label='dBZ')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Deterministic Reflectivity Forecast', fontsize=14)
        plt.tight_layout()
        plt.savefig('forecast_deterministic_refc.png', dpi=150)
        print("Saved plot: forecast_deterministic_refc.png")

if __name__ == '__main__':
    main()