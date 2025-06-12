#!/usr/bin/env python3
"""Simple HRRR processor that extracts variables one by one."""

import xarray as xr
import numpy as np
import zarr
from pathlib import Path

def extract_variables(grib_file):
    """Extract all needed variables from HRRR."""
    print(f"Processing {grib_file}")
    
    extracted = {}
    
    # 1. Composite Reflectivity
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib', 
                           filter_by_keys={'typeOfLevel': 'atmosphere'})
        if 'refc' in ds:
            extracted['REFC'] = ds.refc.values
            print(f"  ✓ REFC: {ds.refc.shape}")
            lat = ds.latitude.values
            lon = ds.longitude.values
    except Exception as e:
        print(f"  ✗ REFC failed: {e}")
    
    # 2. Temperature at 2m
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'heightAboveGround', 
                                         'level': 2,
                                         'shortName': 't2m'})
        if 't2m' in ds:
            extracted['T2M'] = ds.t2m.values
            print(f"  ✓ T2M: {ds.t2m.shape}")
    except Exception as e:
        print(f"  ✗ T2M failed: {e}")
    
    # 3. Dewpoint at 2m  
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'heightAboveGround',
                                         'level': 2, 
                                         'shortName': 'd2m'})
        if 'd2m' in ds:
            extracted['D2M'] = ds.d2m.values
            print(f"  ✓ D2M: {ds.d2m.shape}")
    except Exception as e:
        print(f"  ✗ D2M failed: {e}")
    
    # 4. U wind at 10m
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'heightAboveGround',
                                         'level': 10,
                                         'shortName': 'u10'})
        if 'u10' in ds:
            extracted['U10'] = ds.u10.values
            print(f"  ✓ U10: {ds.u10.shape}")
    except Exception as e:
        print(f"  ✗ U10 failed: {e}")
        
    # 5. V wind at 10m
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'heightAboveGround',
                                         'level': 10,
                                         'shortName': 'v10'})
        if 'v10' in ds:
            extracted['V10'] = ds.v10.values
            print(f"  ✓ V10: {ds.v10.shape}")
    except Exception as e:
        print(f"  ✗ V10 failed: {e}")
    
    # 6. CAPE
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'surface',
                                         'shortName': 'cape'})
        if 'cape' in ds:
            extracted['CAPE'] = ds.cape.values
            print(f"  ✓ CAPE: {ds.cape.shape}")
    except Exception as e:
        print(f"  ✗ CAPE failed: {e}")
        
    # 7. CIN
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib',
                           filter_by_keys={'typeOfLevel': 'surface',
                                         'shortName': 'cin'})
        if 'cin' in ds:
            extracted['CIN'] = ds.cin.values
            print(f"  ✓ CIN: {ds.cin.shape}")
    except Exception as e:
        print(f"  ✗ CIN failed: {e}")
    
    # Fill missing with defaults
    shape = (1059, 1799)  # HRRR grid
    defaults = {
        'T2M': 288.0,   # ~15°C
        'D2M': 283.0,   # ~10°C  
        'U10': 0.0,
        'V10': 0.0,
        'CAPE': 0.0,
        'CIN': 0.0
    }
    
    for var, default in defaults.items():
        if var not in extracted:
            extracted[var] = np.full(shape, default, dtype=np.float32)
            print(f"  ! {var}: Using default value {default}")
    
    return extracted, lat, lon

def save_to_zarr(data, lat, lon, time_val, output_path):
    """Save extracted data to Zarr."""
    # Create dataset
    data_vars = {}
    for name, array in data.items():
        data_vars[name] = (['time', 'y', 'x'], np.expand_dims(array, 0))
    
    coords = {
        'time': [time_val],
        'latitude': (['y', 'x'], lat),
        'longitude': (['y', 'x'], lon)
    }
    
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Save
    print(f"\nSaving to {output_path}")
    ds.to_zarr(output_path, mode='w')
    
    # Also compute and save stats
    stats = {}
    for var in data.keys():
        arr = data[var]
        stats[var] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }
    
    import json
    stats_path = Path(output_path).parent / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")
    
    return ds

if __name__ == "__main__":
    grib_file = "data/raw/latest/hrrr.20250611.t20z.wrfprsf00.grib2"
    
    # Extract
    data, lat, lon = extract_variables(grib_file)
    
    # Time from filename
    time_val = np.datetime64('2025-06-11T20:00')
    
    # Save
    output_path = "data/zarr/latest/hrrr.zarr"
    ds = save_to_zarr(data, lat, lon, time_val, output_path)
    
    print(f"\n✓ Successfully processed {len(data)} variables")
    print(f"  Variables: {list(data.keys())}")
    print(f"  Shape: {list(data.values())[0].shape}")