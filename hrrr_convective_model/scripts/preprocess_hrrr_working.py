#!/usr/bin/env python
"""
Working HRRR GRIB2 to Zarr converter that properly handles HRRR file structure.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import pandas as pd
import numpy as np
import zarr
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def open_hrrr_variables(fp: Path):
    """Open specific variables from HRRR GRIB2 file."""
    print(f"\nProcessing {fp.name}...")
    
    datasets = []
    
    # Define the exact variables we want with their filters
    variable_configs = [
        # Surface variables
        {"name": "TMP_2m", "filter": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}},
        {"name": "DPT_2m", "filter": {"shortName": "2d", "typeOfLevel": "heightAboveGround", "level": 2}},
        {"name": "UGRD_10m", "filter": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}},
        {"name": "VGRD_10m", "filter": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10}},
        
        # Try alternate names
        {"name": "TMP_2m", "filter": {"shortName": "TMP", "typeOfLevel": "heightAboveGround", "level": 2}},
        {"name": "DPT_2m", "filter": {"shortName": "DPT", "typeOfLevel": "heightAboveGround", "level": 2}},
        {"name": "UGRD_10m", "filter": {"shortName": "UGRD", "typeOfLevel": "heightAboveGround", "level": 10}},
        {"name": "VGRD_10m", "filter": {"shortName": "VGRD", "typeOfLevel": "heightAboveGround", "level": 10}},
        
        # Reflectivity and CAPE
        {"name": "REFC", "filter": {"shortName": "REFC", "typeOfLevel": "atmosphere"}},
        {"name": "CAPE", "filter": {"shortName": "CAPE", "typeOfLevel": "surface"}},
        {"name": "CIN", "filter": {"shortName": "CIN", "typeOfLevel": "surface"}},
    ]
    
    found_vars = {}
    
    for var_config in variable_configs:
        try:
            ds = xr.open_dataset(
                fp, 
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": var_config["filter"]}
            )
            
            # Get the variable name (usually the first data var)
            if len(ds.data_vars) > 0:
                orig_name = list(ds.data_vars)[0]
                var_name = var_config["name"].split("_")[0]  # Get base name like TMP, DPT
                
                # Rename to standard name
                ds = ds.rename({orig_name: var_name})
                
                # Handle time coordinate
                if 'valid_time' in ds.coords:
                    ds = ds.rename({'valid_time': 'time'})
                elif 'time' not in ds.coords:
                    # Extract from filename
                    match = pd.Series([fp.name]).str.extract(r'(\d{8})\.t(\d{2})z')[0]
                    if not match.isna().any():
                        date_str, hour_str = match
                        time_val = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}:00")
                        ds = ds.assign_coords(time=time_val)
                
                if var_name not in found_vars:  # Avoid duplicates
                    found_vars[var_name] = ds
                    print(f"  ✓ Found {var_name}")
                    
        except Exception as e:
            continue
    
    if not found_vars:
        print("  ✗ No variables found")
        return None
    
    # Merge all found variables
    try:
        merged = xr.merge(list(found_vars.values()))
        print(f"  Extracted {len(merged.data_vars)} variables: {list(merged.data_vars)}")
        return merged
    except Exception as e:
        print(f"  Error merging: {e}")
        return None

def main(src_dir: Path, out_dir: Path):
    """Main conversion function."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GRIB2 files
    files = sorted(src_dir.glob("*.grib2"))
    if not files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"Found {len(files)} GRIB2 files")
    
    # Process each file
    datasets = []
    for fp in files:
        ds = open_hrrr_variables(fp)
        if ds is not None:
            # Ensure time dimension exists
            if 'time' not in ds.dims and 'time' in ds.coords:
                ds = ds.expand_dims('time')
            datasets.append(ds)
    
    if not datasets:
        raise ValueError("No valid data extracted from GRIB2 files!")
    
    print(f"\nConcatenating {len(datasets)} datasets...")
    
    # Combine all datasets
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        # Find common variables
        common_vars = set(datasets[0].data_vars)
        for ds in datasets[1:]:
            common_vars &= set(ds.data_vars)
        
        print(f"Common variables across all files: {common_vars}")
        
        # Keep only common variables
        datasets_filtered = []
        for ds in datasets:
            datasets_filtered.append(ds[list(common_vars)])
        
        combined = xr.concat(datasets_filtered, dim='time')
        combined = combined.sortby('time')
    
    # Standardize dimension names if needed
    if 'latitude' in combined.dims:
        print("Dimensions: latitude/longitude")
    elif 'y' in combined.dims:
        print("Dimensions: y/x")
        # Optionally rename to latitude/longitude
        # combined = combined.rename({'y': 'latitude', 'x': 'longitude'})
    
    # Write to Zarr
    print("\nWriting to Zarr format...")
    zarr_path = out_dir / "hrrr.zarr"
    
    # Define chunks
    chunks = {'time': 1}
    for dim in combined.dims:
        if dim in ['latitude', 'y']:
            chunks[dim] = min(768, len(combined[dim]))
        elif dim in ['longitude', 'x']:
            chunks[dim] = min(768, len(combined[dim]))
    
    combined = combined.chunk(chunks)
    
    # Write with progress
    print("Writing chunks...")
    combined.to_zarr(zarr_path, mode='w', consolidated=True)
    
    # Write metadata
    meta = {
        "variables": list(combined.data_vars),
        "dimensions": dict(combined.dims),
        "shape": dict(combined.sizes),
        "time_range": {
            "start": str(combined.time.values[0]),
            "end": str(combined.time.values[-1]),
            "steps": len(combined.time)
        }
    }
    
    # Save as both manifest.json and stats.json (placeholder)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Create placeholder stats file
    stats = {}
    for var in combined.data_vars:
        # Quick statistics - in practice you'd compute these properly
        data = combined[var].values
        stats[var] = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)) + 1e-8
        }
    
    with open(out_dir.parent.parent / "data" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Successfully wrote Zarr archive")
    print(f"   Path: {zarr_path}")
    print(f"   Variables: {', '.join(combined.data_vars)}")
    print(f"   Shape: {dict(combined.sizes)}")
    print(f"   Time range: {meta['time_range']['start']} to {meta['time_range']['end']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    
    args = parser.parse_args()
    main(args.src, args.out)