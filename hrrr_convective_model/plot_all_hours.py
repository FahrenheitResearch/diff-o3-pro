#!/usr/bin/env python3
"""Plot all forecast hours."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import zarr
from pathlib import Path

from utils.normalization import Normalizer

# Path to normalization statistics
STATS_PATH = Path('data/stats.json')

# Initialize normalizer if stats file exists
normalizer = Normalizer(STATS_PATH) if STATS_PATH.exists() else None

def denorm(data, var):
    """Apply denormalization if stats are available."""
    if normalizer is None:
        print(f"Warning: No normalizer available for {var}")
        return data
    return normalizer.decode(data, var)

# Load data
print("Loading data...")
# Check if denormalized version exists
denorm_file = Path('forecasts/latest_real/def_forecast_20250612_18Z_denorm.nc')
if denorm_file.exists():
    print("Using denormalized forecast data")
    ds = xr.open_dataset(denorm_file)
else:
    ds = xr.open_dataset('forecasts/latest_real/def_forecast_20250612_18Z.nc')
store = zarr.open('data/zarr/latest/hrrr.zarr', 'r')
lat = store['latitude'][:]
lon = store['longitude'][:]

output_dir = Path('forecasts/realtime/maps')
output_dir.mkdir(parents=True, exist_ok=True)

# Get number of forecast hours
n_hours = len(ds.time)

print(f"Generating individual maps for {n_hours} forecast hours...")

# Variable configurations
var_configs = {
    'REFC': {
        'name': 'Composite Reflectivity',
        'cmap': 'turbo',
        'vmin': -10,
        'vmax': 60,
        'label': 'Reflectivity (dBZ)',
        'mask_below': -10
    },
    'T2M': {
        'name': '2m Temperature',
        'cmap': 'RdBu_r',
        'vmin': -20,
        'vmax': 40,
        'label': 'Temperature (°C)',
        'convert': lambda x: x - 273.15  # K to C
    },
    'CAPE': {
        'name': 'CAPE',
        'cmap': 'hot_r',
        'vmin': 0,
        'vmax': 4000,
        'label': 'CAPE (J/kg)',
        'mask_below': 0
    },
    'wind': {
        'name': '10m Wind Speed',
        'cmap': 'viridis',
        'vmin': 0,
        'vmax': 20,
        'label': 'Wind Speed (m/s)'
    }
}

# Create individual plots for each variable and forecast hour
for t_idx in range(n_hours):
    lead_hour = t_idx + 1
    print(f"\nProcessing hour {lead_hour}...")
    
    # 1. Reflectivity - Ensemble Mean
    if 'REFC' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get ensemble mean
        data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        # Only denormalize if not already denormalized
        if 'normalized' not in ds.attrs or ds.attrs['normalized'] != 'False':
            data = denorm(data, 'REFC')
        
        # Mask out very low reflectivity values (noise)
        cfg = var_configs['REFC']
        if 'mask_below' in cfg:
            data = np.where(data < cfg['mask_below'], np.nan, data)
        
        im = ax.imshow(data, origin='lower', cmap=cfg['cmap'], aspect='auto', 
                      vmin=cfg['vmin'], vmax=cfg['vmax'])
        plt.colorbar(im, ax=ax, label=cfg['label'])
        ax.set_title(f"{cfg['name']} - Ensemble Mean\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f'refc_mean_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # 2. Temperature - Ensemble Mean
    if 'T2M' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get ensemble mean and denormalize
        data = ds.T2M.isel(time=t_idx).mean(dim='member').values
        data = denorm(data, 'T2M')
        
        # Convert K to C
        cfg = var_configs['T2M']
        if 'convert' in cfg:
            data = cfg['convert'](data)
        
        im = ax.imshow(data, origin='lower', cmap=cfg['cmap'], aspect='auto',
                      vmin=cfg['vmin'], vmax=cfg['vmax'])
        plt.colorbar(im, ax=ax, label=cfg['label'])
        ax.set_title(f"{cfg['name']} - Ensemble Mean\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f't2m_mean_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # 3. CAPE - Ensemble Mean
    if 'CAPE' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get ensemble mean and denormalize
        data = ds.CAPE.isel(time=t_idx).mean(dim='member').values
        data = denorm(data, 'CAPE')
        
        # Mask negative values
        cfg = var_configs['CAPE']
        if 'mask_below' in cfg:
            data = np.where(data < cfg['mask_below'], np.nan, data)
        
        im = ax.imshow(data, origin='lower', cmap=cfg['cmap'], aspect='auto',
                      vmin=cfg['vmin'], vmax=cfg['vmax'])
        plt.colorbar(im, ax=ax, label=cfg['label'])
        ax.set_title(f"{cfg['name']} - Ensemble Mean\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f'cape_mean_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # 4. Wind Speed - Ensemble Mean
    if 'U10' in ds and 'V10' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get ensemble mean and denormalize
        u = ds.U10.isel(time=t_idx).mean(dim='member').values
        v = ds.V10.isel(time=t_idx).mean(dim='member').values
        u = denorm(u, 'U10')
        v = denorm(v, 'V10')
        wspd = np.sqrt(u**2 + v**2)
        
        cfg = var_configs['wind']
        im = ax.imshow(wspd, origin='lower', cmap=cfg['cmap'], aspect='auto',
                      vmin=cfg['vmin'], vmax=cfg['vmax'])
        plt.colorbar(im, ax=ax, label=cfg['label'])
        ax.set_title(f"{cfg['name']} - Ensemble Mean\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f'wind_mean_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # 5. Reflectivity - Ensemble Spread
    if 'REFC' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get all members and denormalize each one
        members = ds.REFC.isel(time=t_idx).values
        members_denorm = np.array([denorm(members[i], 'REFC') for i in range(members.shape[0])])
        data = np.std(members_denorm, axis=0)
        
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0, vmax=15)
        plt.colorbar(im, ax=ax, label='Std Dev (dBZ)')
        ax.set_title(f"Reflectivity - Ensemble Spread\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f'refc_spread_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    # 6. Temperature - Ensemble Spread
    if 'T2M' in ds:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get all members and denormalize each one
        members = ds.T2M.isel(time=t_idx).values
        members_denorm = np.array([denorm(members[i], 'T2M') - 273.15 for i in range(members.shape[0])])
        data = np.std(members_denorm, axis=0)
        
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0, vmax=5)
        plt.colorbar(im, ax=ax, label='Std Dev (°C)')
        ax.set_title(f"Temperature - Ensemble Spread\nForecast Hour +{lead_hour}", fontsize=14)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        output_file = output_dir / f't2m_spread_f{lead_hour:03d}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

# Create a simple reflectivity evolution plot (smaller)
print("\nCreating reflectivity evolution plot...")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for t_idx in range(min(6, n_hours)):
    ax = axes[t_idx]
    if 'REFC' in ds:
        data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        data = denorm(data, 'REFC')
        # Mask out very low reflectivity values (noise)
        data = np.where(data < -10, np.nan, data)
        im = ax.imshow(data, origin='lower', cmap='turbo', aspect='auto', vmin=-10, vmax=60)
        # Smaller colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('dBZ', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    ax.set_title(f'Hour +{t_idx + 1}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Reflectivity Evolution (First 6 Hours)', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'refc_evolution_6hr.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved evolution plot: {output_dir / 'refc_evolution_6hr.png'}")

print(f"\nAll individual visualizations saved to: {output_dir}/")
print("Files created:")
for f in sorted(output_dir.glob('*.png')):
    # Get file size in MB
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  - {f.name} ({size_mb:.2f} MB)")