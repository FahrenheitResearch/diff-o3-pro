#!/usr/bin/env python3
"""Denormalize forecast data from z-scores to physical units."""

import xarray as xr
import numpy as np
from pathlib import Path
from utils.normalization import Normalizer

def denormalize_forecast(input_file, output_file, stats_file):
    """Denormalize forecast data."""
    print(f"Loading forecast from: {input_file}")
    ds = xr.open_dataset(input_file)
    
    # Initialize normalizer
    normalizer = Normalizer(Path(stats_file))
    
    # Create a new dataset for denormalized data
    ds_denorm = ds.copy()
    
    # List of variables to denormalize
    variables = ['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN']
    
    print("Denormalizing variables...")
    for var in variables:
        # Check for regular variable
        if var in ds:
            print(f"  {var}: ", end='')
            data = ds[var].values
            # Denormalize each ensemble member and time step
            denorm_data = np.zeros_like(data)
            for m in range(data.shape[0]):  # members
                for t in range(data.shape[1]):  # time
                    denorm_data[m, t] = normalizer.decode(data[m, t], var)
            ds_denorm[var] = (ds[var].dims, denorm_data)
            print(f"range [{np.nanmin(denorm_data):.2f}, {np.nanmax(denorm_data):.2f}]")
        
        # Check for mean variable
        if f"{var}_mean" in ds:
            print(f"  {var}_mean: ", end='')
            data = ds[f"{var}_mean"].values
            # Denormalize each time step
            denorm_data = np.zeros_like(data)
            for t in range(data.shape[0]):  # time
                denorm_data[t] = normalizer.decode(data[t], var)
            ds_denorm[f"{var}_mean"] = (ds[f"{var}_mean"].dims, denorm_data)
            print(f"range [{np.nanmin(denorm_data):.2f}, {np.nanmax(denorm_data):.2f}]")
        
        # For spread, we need special handling
        if f"{var}_spread" in ds:
            print(f"  {var}_spread: ", end='')
            # Spread is in normalized units, need to scale by std
            stats = normalizer.stats[var]
            std = stats['std']
            data = ds[f"{var}_spread"].values
            # Spread in physical units = spread in z-scores * std
            denorm_data = data * std
            ds_denorm[f"{var}_spread"] = (ds[f"{var}_spread"].dims, denorm_data)
            print(f"range [{np.nanmin(denorm_data):.2f}, {np.nanmax(denorm_data):.2f}]")
    
    # Update attributes
    ds_denorm.attrs['normalized'] = 'False'
    ds_denorm.attrs['denormalized_by'] = 'denormalize_forecast.py'
    
    # Save denormalized forecast
    print(f"\nSaving denormalized forecast to: {output_file}")
    ds_denorm.to_netcdf(output_file)
    print("Done!")
    
    return ds_denorm

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Denormalize forecast data')
    parser.add_argument('input', type=str, help='Input forecast file')
    parser.add_argument('--output', type=str, help='Output file (default: adds _denorm suffix)')
    parser.add_argument('--stats', type=str, default='data/stats.json', help='Stats file')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_denorm{input_file.suffix}"
    
    denormalize_forecast(input_file, output_file, args.stats)