#!/usr/bin/env python3
"""Process latest HRRR data with proper level filtering."""

import xarray as xr
import zarr
import numpy as np
from pathlib import Path
import pandas as pd

# Variables we need
VARIABLES = {
    'REFC': {'typeOfLevel': 'atmosphere', 'level': 0},  # Composite reflectivity
    't2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},  # 2m temperature
    'd2m': {'typeOfLevel': 'heightAboveGround', 'level': 2},  # 2m dewpoint
    'u10': {'typeOfLevel': 'heightAboveGround', 'level': 10}, # 10m U wind
    'v10': {'typeOfLevel': 'heightAboveGround', 'level': 10}, # 10m V wind
    'cape': {'typeOfLevel': 'surface', 'level': 0, 'stepType': 'instant'},  # CAPE
    'cin': {'typeOfLevel': 'surface', 'level': 0, 'stepType': 'instant'},   # CIN
}

# Output variable names
VAR_MAPPING = {
    't2m': 'T2M',
    'd2m': 'D2M', 
    'u10': 'U10',
    'v10': 'V10',
    'cape': 'CAPE',
    'cin': 'CIN',
    'REFC': 'REFC'
}

def process_hrrr(grib_file, output_dir):
    """Process single HRRR file to Zarr."""
    print(f"Processing {grib_file}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract each variable with proper filtering
    datasets = []
    
    for var_name, filters in VARIABLES.items():
        try:
            print(f"  Extracting {var_name}...")
            ds = xr.open_dataset(
                grib_file,
                engine='cfgrib',
                filter_by_keys=filters
            )
            
            # Find the actual variable name in the dataset
            actual_var = None
            for v in ds.data_vars:
                if var_name in str(v).upper():
                    actual_var = v
                    break
            
            if actual_var:
                # Rename to standard name
                output_name = VAR_MAPPING.get(var_name, var_name)
                data = ds[actual_var].values
                print(f"    ✓ Found {actual_var} -> {output_name}, shape: {data.shape}")
                
                # Store with consistent naming
                datasets.append({
                    'name': output_name,
                    'data': data,
                    'attrs': ds[actual_var].attrs
                })
        except Exception as e:
            print(f"    ✗ Failed to extract {var_name}: {e}")
    
    if not datasets:
        raise ValueError("No variables extracted!")
    
    # Get coordinates from first successful dataset
    ds_ref = xr.open_dataset(grib_file, engine='cfgrib', 
                           filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
    
    # Create output dataset
    print("\nCreating Zarr dataset...")
    
    # Time from filename
    import re
    match = re.search(r'(\d{8})\.t(\d{2})z', Path(grib_file).name)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        time_val = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{hour_str}:00")
    else:
        time_val = np.datetime64('now')
    
    # Create dataset
    data_vars = {}
    for item in datasets:
        data_vars[item['name']] = (['time', 'y', 'x'], 
                                  np.expand_dims(item['data'], 0))
    
    coords = {
        'time': [time_val],
        'latitude': (['y', 'x'], ds_ref.latitude.values),
        'longitude': (['y', 'x'], ds_ref.longitude.values)
    }
    
    ds_out = xr.Dataset(data_vars, coords=coords)
    
    # Save to Zarr
    zarr_path = output_dir / 'hrrr.zarr'
    print(f"Writing to {zarr_path}")
    ds_out.to_zarr(zarr_path, mode='w')
    
    print(f"\n✓ Successfully processed {len(datasets)} variables")
    print(f"  Time: {time_val}")
    print(f"  Grid: {ds_ref.latitude.shape}")
    
    return zarr_path

if __name__ == "__main__":
    grib_file = "data/raw/latest/hrrr.20250611.t20z.wrfprsf00.grib2"
    output_dir = "data/zarr/latest"
    
    zarr_path = process_hrrr(grib_file, output_dir)
    print(f"\nOutput saved to: {zarr_path}")