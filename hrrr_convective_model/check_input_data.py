#!/usr/bin/env python3
"""Check HRRR input data quality."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import zarr
from pathlib import Path
import json

# Load the input data
print("Loading HRRR input data...")
store = zarr.open('data/zarr/latest/hrrr.zarr', 'r')

# Check available variables
print("\nAvailable variables in zarr store:")
for key in store.keys():
    if key not in ['latitude', 'longitude', 'time']:
        print(f"  - {key}: shape = {store[key].shape}")

# Get dimensions
print(f"\nData dimensions:")
print(f"  - time: {store['time'].shape}")
print(f"  - latitude: {store['latitude'].shape}")
print(f"  - longitude: {store['longitude'].shape}")

# Check time values
time_data = store['time'][:]
print(f"\nTime values: {time_data}")

# Load normalization stats to understand expected ranges
stats_path = Path('data/stats.json')
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    print("\nNormalization statistics:")
    for var in ['REFC', 'T2M', 'U10', 'V10', 'CAPE']:
        if var in stats:
            print(f"\n  {var}:")
            print(f"    mean: {stats[var]['mean']:.2f}")
            print(f"    std: {stats[var]['std']:.2f}")
            print(f"    min: {stats[var]['min']:.2f}")
            print(f"    max: {stats[var]['max']:.2f}")

# Check the actual data at time index 0
print("\n\nChecking input data at time index 0:")

# Check key variables
variables = ['REFC', 'T2M', 'U10', 'V10', 'CAPE']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    if var in store:
        data = store[var][0]  # First time step
        ax = axes[i]
        
        # Print statistics
        print(f"\n{var} statistics:")
        print(f"  Shape: {data.shape}")
        print(f"  Min: {np.nanmin(data):.2f}")
        print(f"  Max: {np.nanmax(data):.2f}")
        print(f"  Mean: {np.nanmean(data):.2f}")
        print(f"  Std: {np.nanstd(data):.2f}")
        print(f"  % NaN: {100 * np.isnan(data).sum() / data.size:.2f}%")
        
        # Plot
        if var == 'REFC':
            # Mask low reflectivity
            data_plot = np.where(data < -10, np.nan, data)
            im = ax.imshow(data_plot, origin='lower', cmap='turbo', vmin=-10, vmax=60)
            ax.set_title(f'{var} (dBZ)')
        elif var == 'T2M':
            # Convert K to C
            data_plot = data - 273.15
            im = ax.imshow(data_plot, origin='lower', cmap='RdBu_r', vmin=-20, vmax=40)
            ax.set_title(f'{var} (°C)')
        elif var == 'CAPE':
            # Mask negative CAPE
            data_plot = np.where(data < 0, np.nan, data)
            im = ax.imshow(data_plot, origin='lower', cmap='hot_r', vmin=0, vmax=4000)
            ax.set_title(f'{var} (J/kg)')
        else:
            im = ax.imshow(data, origin='lower', cmap='viridis')
            ax.set_title(f'{var}')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')

# Wind speed
if 'U10' in store and 'V10' in store:
    u = store['U10'][0]
    v = store['V10'][0]
    wspd = np.sqrt(u**2 + v**2)
    
    ax = axes[5]
    im = ax.imshow(wspd, origin='lower', cmap='viridis', vmin=0, vmax=20)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('10m Wind Speed (m/s)')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    
    print(f"\nWind Speed statistics:")
    print(f"  Min: {np.nanmin(wspd):.2f}")
    print(f"  Max: {np.nanmax(wspd):.2f}")
    print(f"  Mean: {np.nanmean(wspd):.2f}")

plt.suptitle('HRRR Input Data at Time 0', fontsize=14)
plt.tight_layout()
plt.savefig('input_data_check.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: input_data_check.png")

# Compare with forecast output
print("\n\nComparing with forecast output:")
forecast_file = 'forecasts/latest_real/def_forecast_20250612_18Z.nc'
if Path(forecast_file).exists():
    ds = xr.open_dataset(forecast_file)
    print(f"\nForecast dimensions:")
    print(f"  time: {len(ds.time)}")
    print(f"  member: {len(ds.member)}")
    print(f"  y: {len(ds.y)}")
    print(f"  x: {len(ds.x)}")
    
    # Check output statistics at first hour
    print("\nForecast output statistics at hour 1:")
    for var in ['REFC', 'T2M', 'U10', 'V10', 'CAPE']:
        if var in ds:
            data = ds[var].isel(time=0, member=0).values
            print(f"\n{var}:")
            print(f"  Min: {np.nanmin(data):.2f}")
            print(f"  Max: {np.nanmax(data):.2f}")
            print(f"  Mean: {np.nanmean(data):.2f}")
            print(f"  Std: {np.nanstd(data):.2f}")