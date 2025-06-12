#!/usr/bin/env python3
"""Plot validation forecast for all hours."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import zarr
from pathlib import Path
import sys

# File to plot
forecast_file = Path('forecasts/validation/def_forecast_20250101_00Z.nc')

if not forecast_file.exists():
    print(f"Error: Forecast file not found: {forecast_file}")
    sys.exit(1)

# Load data
print(f"Loading forecast data from: {forecast_file}")
ds = xr.open_dataset(forecast_file)

# Check if zarr data exists for lat/lon
zarr_path = Path('data/zarr/training_data/hrrr.zarr')
if zarr_path.exists():
    print("Loading coordinate data from zarr...")
    store = zarr.open(zarr_path, 'r')
    if 'latitude' in store and 'longitude' in store:
        lat = store['latitude'][:]
        lon = store['longitude'][:]
    else:
        lat, lon = None, None
else:
    print("Warning: Zarr coordinate data not found")
    lat, lon = None, None

output_dir = Path('forecasts/validation/maps')
output_dir.mkdir(parents=True, exist_ok=True)

# Get dataset info
print("\nDataset info:")
print(f"  Variables: {list(ds.data_vars)}")
print(f"  Dimensions: {dict(ds.dims)}")
if 'time' in ds.dims:
    n_hours = len(ds.time)
    print(f"  Forecast hours: {n_hours}")
else:
    print("Warning: No time dimension found")
    n_hours = 1

if 'member' in ds.dims:
    n_members = len(ds.member)
    print(f"  Ensemble members: {n_members}")
else:
    n_members = 1

print(f"\nGenerating maps for {min(n_hours, 6)} forecast hours...")

# Create a figure for each forecast hour
for t_idx in range(min(n_hours, 6)):
    print(f"  Processing hour {t_idx + 1}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Reflectivity - Ensemble Mean
    ax = axes[0]
    if 'REFC' in ds:
        if 'member' in ds.REFC.dims:
            data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        else:
            data = ds.REFC.isel(time=t_idx).values if 'time' in ds.REFC.dims else ds.REFC.values
        im = ax.imshow(data, origin='lower', cmap='turbo', aspect='auto', vmin=0, vmax=50)
        plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)')
    ax.set_title('Composite Reflectivity - Ensemble Mean')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 2. Temperature - Ensemble Mean
    ax = axes[1]
    if 'T2M' in ds:
        if 'member' in ds.T2M.dims:
            data = ds.T2M.isel(time=t_idx).mean(dim='member').values
        else:
            data = ds.T2M.isel(time=t_idx).values if 'time' in ds.T2M.dims else ds.T2M.values
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='T2M (normalized)')
    ax.set_title('2m Temperature - Ensemble Mean')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 3. CAPE - Ensemble Mean
    ax = axes[2]
    if 'CAPE' in ds:
        if 'member' in ds.CAPE.dims:
            data = ds.CAPE.isel(time=t_idx).mean(dim='member').values
        else:
            data = ds.CAPE.isel(time=t_idx).values if 'time' in ds.CAPE.dims else ds.CAPE.values
        im = ax.imshow(data, origin='lower', cmap='hot_r', aspect='auto', vmin=-1, vmax=2)
        plt.colorbar(im, ax=ax, label='CAPE (normalized)')
    ax.set_title('CAPE - Ensemble Mean')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 4. Wind Speed - Ensemble Mean
    ax = axes[3]
    if 'U10' in ds and 'V10' in ds:
        if 'member' in ds.U10.dims:
            u = ds.U10.isel(time=t_idx).mean(dim='member').values
            v = ds.V10.isel(time=t_idx).mean(dim='member').values
        else:
            u = ds.U10.isel(time=t_idx).values if 'time' in ds.U10.dims else ds.U10.values
            v = ds.V10.isel(time=t_idx).values if 'time' in ds.V10.dims else ds.V10.values
        wspd = np.sqrt(u**2 + v**2)
        im = ax.imshow(wspd, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Wind Speed (normalized)')
    ax.set_title('10m Wind Speed - Ensemble Mean')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 5. Reflectivity - Ensemble Spread (if ensemble data)
    ax = axes[4]
    if 'REFC' in ds and 'member' in ds.REFC.dims and n_members > 1:
        data = ds.REFC.isel(time=t_idx).std(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Std Dev')
        ax.set_title('Reflectivity - Ensemble Spread')
    else:
        ax.text(0.5, 0.5, 'N/A\n(No ensemble)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reflectivity - Ensemble Spread')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 6. Temperature - Ensemble Spread (if ensemble data)
    ax = axes[5]
    if 'T2M' in ds and 'member' in ds.T2M.dims and n_members > 1:
        data = ds.T2M.isel(time=t_idx).std(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Std Dev')
        ax.set_title('Temperature - Ensemble Spread')
    else:
        ax.text(0.5, 0.5, 'N/A\n(No ensemble)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Temperature - Ensemble Spread')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add main title
    lead_hour = t_idx + 1
    plt.suptitle(f'DEF Validation Forecast (2025-01-01 00Z) - Lead Time: +{lead_hour} hour{"s" if lead_hour > 1 else ""}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'validation_forecast_f{lead_hour:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")

# Also create an animated GIF
print("\nCreating animated GIF...")
import subprocess

# Get all forecast images
image_files = sorted(output_dir.glob('validation_forecast_f*.png'))
if image_files:
    # Create GIF using ImageMagick if available
    try:
        cmd = ['convert', '-delay', '100', '-loop', '0'] + [str(f) for f in image_files] + [str(output_dir / 'validation_forecast_animation.gif')]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Created animation: {output_dir / 'validation_forecast_animation.gif'}")
    except:
        print("Could not create GIF (ImageMagick not installed)")

# Create a summary plot with all hours
print("\nCreating summary plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for t_idx in range(min(6, n_hours)):
    ax = axes[t_idx]
    if 'REFC' in ds:
        if 'member' in ds.REFC.dims:
            data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        else:
            data = ds.REFC.isel(time=t_idx).values if 'time' in ds.REFC.dims else ds.REFC.values
        im = ax.imshow(data, origin='lower', cmap='turbo', aspect='auto', vmin=0, vmax=50)
        plt.colorbar(im, ax=ax, label='dBZ', shrink=0.6)
    ax.set_title(f'Hour +{t_idx + 1}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('DEF Validation Forecast - Reflectivity Evolution (2025-01-01 00Z)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'validation_forecast_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary: {output_dir / 'validation_forecast_summary.png'}")

print(f"\nAll visualizations saved to: {output_dir}/")
print("Files created:")
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")

# Close dataset
ds.close()