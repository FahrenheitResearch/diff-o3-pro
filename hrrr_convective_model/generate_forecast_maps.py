#!/usr/bin/env python3
"""Generate real forecast maps with visualization."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime, timedelta
import zarr
import torch
import yaml
import argparse
from tqdm import tqdm

# Import necessary modules
from models.unet_attention_fixed import UNetAttn
from models.diffusion.ddpm_conditioned import ConditionalDiffusionUNet, GaussianDiffusion
from utils.normalization import Normalizer

# HRRR domain bounds
HRRR_BOUNDS = {
    'west': -134.0,
    'east': -60.0,
    'south': 21.0,
    'north': 53.0
}

# Custom colormaps for weather variables
def get_precip_colormap():
    """Create a nice colormap for precipitation/reflectivity."""
    colors = [
        (1.0, 1.0, 1.0, 0.0),  # Transparent white
        (0.0, 1.0, 1.0, 0.8),  # Cyan
        (0.0, 0.5, 1.0, 0.8),  # Blue
        (0.0, 1.0, 0.0, 0.8),  # Green
        (1.0, 1.0, 0.0, 0.8),  # Yellow
        (1.0, 0.5, 0.0, 0.8),  # Orange
        (1.0, 0.0, 0.0, 0.8),  # Red
        (1.0, 0.0, 1.0, 0.8),  # Magenta
    ]
    cmap = LinearSegmentedColormap.from_list('precip', colors)
    return cmap

def get_temp_colormap():
    """Create a nice colormap for temperature."""
    return plt.cm.RdBu_r

def plot_forecast_panel(ax, data, var_name, title, lat, lon):
    """Plot a single forecast panel."""
    ax.set_extent([HRRR_BOUNDS['west'], HRRR_BOUNDS['east'], 
                   HRRR_BOUNDS['south'], HRRR_BOUNDS['north']], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
    
    # Choose colormap and levels based on variable
    if var_name == 'REFC':
        cmap = get_precip_colormap()
        levels = np.arange(0, 75, 5)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    elif var_name == 'T2M':
        cmap = get_temp_colormap()
        # Convert from K to F
        data = (data - 273.15) * 9/5 + 32
        levels = np.arange(0, 110, 5)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    elif var_name == 'CAPE':
        cmap = plt.cm.hot_r
        levels = [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    else:
        cmap = plt.cm.viridis
        norm = None
    
    # Plot data
    im = ax.pcolormesh(lon, lat, data, 
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm,
                       shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                        pad=0.05, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    # Set title
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    return im

def generate_forecast_maps(forecast_path, output_dir, zarr_path):
    """Generate visualization maps from forecast NetCDF."""
    # Load forecast
    print(f"Loading forecast from {forecast_path}")
    ds = xr.open_dataset(forecast_path)
    
    # Load lat/lon from zarr
    store = zarr.open(str(zarr_path), 'r')
    lat = store['latitude'][:]
    lon = store['longitude'][:]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Variables to plot
    plot_vars = ['REFC', 'T2M', 'CAPE']
    
    # Get time info
    init_time = pd.to_datetime(ds.attrs.get('initialization_time', '2025-01-01 00:00:00'))
    
    # Plot each lead time
    for t_idx, lead_hour in enumerate(ds.lead_time.values):
        print(f"Plotting lead hour {lead_hour}")
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Get valid time
        valid_time = init_time + pd.Timedelta(hours=int(lead_hour))
        
        # Main title
        fig.suptitle(f'DEF Ensemble Forecast - Valid: {valid_time:%Y-%m-%d %H}Z\n' + 
                     f'Init: {init_time:%Y-%m-%d %H}Z | Lead: {lead_hour}h',
                     fontsize=14, fontweight='bold')
        
        # Plot each variable
        for v_idx, var in enumerate(plot_vars):
            if var not in ds:
                continue
                
            # Ensemble mean
            ax1 = fig.add_subplot(3, 3, v_idx*3 + 1, projection=ccrs.PlateCarree())
            data_mean = ds[var].isel(lead_time=t_idx).mean(dim='member')
            plot_forecast_panel(ax1, data_mean, var, f'{var} - Ensemble Mean', lat, lon)
            
            # Ensemble spread
            ax2 = fig.add_subplot(3, 3, v_idx*3 + 2, projection=ccrs.PlateCarree())
            data_spread = ds[var].isel(lead_time=t_idx).std(dim='member')
            plot_forecast_panel(ax2, data_spread, var, f'{var} - Ensemble Spread', lat, lon)
            
            # Member 1 (deterministic)
            ax3 = fig.add_subplot(3, 3, v_idx*3 + 3, projection=ccrs.PlateCarree())
            data_m1 = ds[var].isel(lead_time=t_idx, member=0)
            plot_forecast_panel(ax3, data_m1, var, f'{var} - Member 1', lat, lon)
        
        # Save figure
        output_file = output_dir / f'forecast_{init_time:%Y%m%d_%H}Z_f{lead_hour:03d}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {output_file}")

def run_real_forecast():
    """Run a real forecast using the trained models."""
    # Configuration
    device = torch.device('cuda')
    config_path = 'configs/test.yaml'
    zarr_path = Path('data/zarr/test/hrrr.zarr')
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override settings for real forecast
    cfg['ensemble']['num_members'] = 4
    cfg['diffusion']['num_steps'] = 10  # Reduced for speed
    
    # Get current time for forecast
    forecast_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecast_time_str = forecast_time.strftime("%Y-%m-%d %H")
    
    print(f"Generating forecast for: {forecast_time_str}")
    
    # Run inference
    import subprocess
    cmd = [
        'python', 'inference_ensemble.py',
        '--config', config_path,
        '--start-date', forecast_time_str,
        '--cycles', '1',
        '--max-lead-hours', '6',  # 6 hour forecast
        '--ensemble-size', '4',
        '--device', 'cuda',  # Use GPU!
        '--output-dir', 'forecasts/realtime'
    ]
    
    print("Running ensemble inference on GPU...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running inference:")
        print(result.stderr)
        return
    
    print(result.stdout)
    
    # Find the generated forecast file
    forecast_dir = Path('forecasts/realtime')
    forecast_files = list(forecast_dir.glob('*.nc'))
    if not forecast_files:
        print("No forecast files generated!")
        return
        
    latest_forecast = max(forecast_files, key=lambda p: p.stat().st_mtime)
    print(f"Found forecast: {latest_forecast}")
    
    # Generate maps
    output_dir = Path('forecasts/realtime/maps')
    generate_forecast_maps(latest_forecast, output_dir, zarr_path)
    
    print(f"\nMaps saved to: {output_dir}")
    print("Done!")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues
    run_real_forecast()