#!/usr/bin/env python3
"""Plot all forecast hours."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import zarr
from pathlib import Path

# Load data
print("Loading data...")
ds = xr.open_dataset('forecasts/realtime/def_forecast_20250611_20Z.nc')
store = zarr.open('data/zarr/latest/hrrr.zarr', 'r')
lat = store['latitude'][:]
lon = store['longitude'][:]

output_dir = Path('forecasts/realtime/maps')
output_dir.mkdir(parents=True, exist_ok=True)

# Get number of forecast hours
n_hours = min(len(ds.time), 6)  # Plot up to 6 hours

print(f"Generating maps for {n_hours} forecast hours...")

# Create a figure for each forecast hour
for t_idx in range(n_hours):
    print(f"  Processing hour {t_idx + 1}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Reflectivity - Ensemble Mean
    ax = axes[0]
    if 'REFC' in ds:
        data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='turbo', aspect='auto', vmin=0, vmax=50)
        plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)')
    ax.set_title('Composite Reflectivity - Ensemble Mean')
    
    # 2. Temperature - Ensemble Mean
    ax = axes[1]
    if 'T2M' in ds:
        data = ds.T2M.isel(time=t_idx).mean(dim='member').values
        # Note: Data is normalized, would need denormalization for real temps
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='T2M (normalized)')
    ax.set_title('2m Temperature - Ensemble Mean')
    
    # 3. CAPE - Ensemble Mean
    ax = axes[2]
    if 'CAPE' in ds:
        data = ds.CAPE.isel(time=t_idx).mean(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='hot_r', aspect='auto', vmin=-1, vmax=2)
        plt.colorbar(im, ax=ax, label='CAPE (normalized)')
    ax.set_title('CAPE - Ensemble Mean')
    
    # 4. Wind Speed - Ensemble Mean
    ax = axes[3]
    if 'U10' in ds and 'V10' in ds:
        u = ds.U10.isel(time=t_idx).mean(dim='member').values
        v = ds.V10.isel(time=t_idx).mean(dim='member').values
        wspd = np.sqrt(u**2 + v**2)
        im = ax.imshow(wspd, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Wind Speed (normalized)')
    ax.set_title('10m Wind Speed - Ensemble Mean')
    
    # 5. Reflectivity - Ensemble Spread
    ax = axes[4]
    if 'REFC' in ds:
        data = ds.REFC.isel(time=t_idx).std(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Std Dev')
    ax.set_title('Reflectivity - Ensemble Spread')
    
    # 6. Temperature - Ensemble Spread
    ax = axes[5]
    if 'T2M' in ds:
        data = ds.T2M.isel(time=t_idx).std(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Std Dev')
    ax.set_title('Temperature - Ensemble Spread')
    
    # Add main title
    lead_hour = t_idx + 1
    plt.suptitle(f'DEF Ensemble Forecast - Lead Time: +{lead_hour} hour{"s" if lead_hour > 1 else ""}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'forecast_f{lead_hour:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")

# Also create an animated GIF
print("\nCreating animated GIF...")
import subprocess

# Get all forecast images
image_files = sorted(output_dir.glob('forecast_f*.png'))
if image_files:
    # Create GIF using ImageMagick if available
    try:
        cmd = ['convert', '-delay', '100', '-loop', '0'] + [str(f) for f in image_files] + [str(output_dir / 'forecast_animation.gif')]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Created animation: {output_dir / 'forecast_animation.gif'}")
    except:
        print("Could not create GIF (ImageMagick not installed)")

# Create a summary plot with all hours
print("\nCreating summary plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for t_idx in range(min(6, n_hours)):
    ax = axes[t_idx]
    if 'REFC' in ds:
        data = ds.REFC.isel(time=t_idx).mean(dim='member').values
        im = ax.imshow(data, origin='lower', cmap='turbo', aspect='auto', vmin=0, vmax=50)
        plt.colorbar(im, ax=ax, label='dBZ', shrink=0.6)
    ax.set_title(f'Hour +{t_idx + 1}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('DEF Ensemble Forecast - Reflectivity Evolution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'forecast_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved summary: {output_dir / 'forecast_summary.png'}")

print(f"\nAll visualizations saved to: {output_dir}/")
print("Files created:")
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")