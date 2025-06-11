#!/usr/bin/env python
"""
Simple GRIB2 to Zarr converter without Dask for better compatibility.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import pandas as pd
import zarr
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Simplified variable list - start with most common ones
VARS = {
    "t2m": {"typeOfLevel": "heightAboveGround", "level": 2},  # 2m temperature
    "d2m": {"typeOfLevel": "heightAboveGround", "level": 2},  # 2m dewpoint
    "u10": {"typeOfLevel": "heightAboveGround", "level": 10}, # 10m u-wind
    "v10": {"typeOfLevel": "heightAboveGround", "level": 10}, # 10m v-wind
    "refc": {},  # Composite reflectivity
}

def open_grib_file(fp: Path):
    """Open GRIB2 file and extract available variables."""
    print(f"\nProcessing {fp.name}...")
    
    # First, try to open without filters to see what's available
    try:
        ds_all = xr.open_dataset(fp, engine="cfgrib")
        print(f"  Available variables: {list(ds_all.data_vars)}")
        print(f"  Coordinates: {list(ds_all.coords)}")
        
        # Extract time
        if 'time' in ds_all.coords:
            time_val = ds_all.time.values
        elif 'valid_time' in ds_all.coords:
            time_val = ds_all.valid_time.values
        else:
            # Try to extract from filename
            import re
            match = re.search(r'(\d{8})\.t(\d{2})z', fp.name)
            if match:
                date_str = match.group(1)
                hour_str = match.group(2)
                time_val = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}:00")
            else:
                print("  Warning: Could not determine time")
                time_val = None
        
        # Create a simplified dataset with common variables
        data_vars = {}
        coords = {'time': time_val} if time_val is not None else {}
        
        # Get dimensions from first variable
        first_var = list(ds_all.data_vars)[0]
        if 'latitude' in ds_all[first_var].coords:
            coords['latitude'] = ds_all.latitude
            coords['longitude'] = ds_all.longitude
        elif 'lat' in ds_all[first_var].coords:
            coords['lat'] = ds_all.lat
            coords['lon'] = ds_all.lon
        
        # Extract variables we care about
        for var in ds_all.data_vars:
            if var in ['t2m', 't', 'TMP', '2t']:  # Temperature variants
                data_vars['TMP'] = ds_all[var]
            elif var in ['d2m', 'd', 'DPT', '2d']:  # Dewpoint variants
                data_vars['DPT'] = ds_all[var]
            elif var in ['u10', 'u', 'UGRD', '10u']:  # U-wind variants
                data_vars['UGRD'] = ds_all[var]
            elif var in ['v10', 'v', 'VGRD', '10v']:  # V-wind variants
                data_vars['VGRD'] = ds_all[var]
            elif var in ['refc', 'REFC']:  # Reflectivity
                data_vars['REFC'] = ds_all[var]
        
        if data_vars:
            ds = xr.Dataset(data_vars, coords=coords)
            print(f"  Extracted {len(data_vars)} variables: {list(data_vars.keys())}")
            return ds
        else:
            print("  No matching variables found")
            return None
            
    except Exception as e:
        print(f"  Error opening file: {e}")
        return None

def main(src_dir: Path, out_dir: Path):
    """Main conversion function."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GRIB2 files
    files = sorted(src_dir.glob("*.grib2"))
    if not files:
        files = sorted(src_dir.glob("*.grb2"))
    
    if not files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"Found {len(files)} GRIB2 files")
    
    # Process each file
    datasets = []
    for fp in files:
        ds = open_grib_file(fp)
        if ds is not None:
            datasets.append(ds)
    
    if not datasets:
        raise ValueError("No valid data extracted from GRIB2 files!")
    
    print(f"\nConcatenating {len(datasets)} datasets...")
    
    # Combine all datasets
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        # Ensure all have time dimension
        for ds in datasets:
            if 'time' not in ds.dims:
                ds = ds.expand_dims('time')
        
        combined = xr.concat(datasets, dim='time', data_vars='minimal')
        combined = combined.sortby('time')
    
    # Write to Zarr
    print("\nWriting to Zarr format...")
    zarr_path = out_dir / "hrrr.zarr"
    
    # Simple chunking
    chunks = {'time': 1}
    if 'latitude' in combined.dims:
        chunks['latitude'] = min(768, len(combined.latitude))
        chunks['longitude'] = min(768, len(combined.longitude))
    elif 'lat' in combined.dims:
        chunks['lat'] = min(768, len(combined.lat))
        chunks['lon'] = min(768, len(combined.lon))
    
    combined = combined.chunk(chunks)
    combined.to_zarr(zarr_path, mode='w', consolidated=True)
    
    # Write metadata
    meta = {
        "variables": list(combined.data_vars),
        "dimensions": dict(combined.dims),
        "shape": dict(combined.sizes),
        "time_steps": len(combined.time) if 'time' in combined.dims else 1
    }
    
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Successfully wrote Zarr archive")
    print(f"   Path: {zarr_path}")
    print(f"   Variables: {', '.join(combined.data_vars)}")
    print(f"   Shape: {dict(combined.sizes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="Source directory")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    
    args = parser.parse_args()
    main(args.src, args.out)