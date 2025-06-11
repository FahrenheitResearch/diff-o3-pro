#!/usr/bin/env python
"""
Robust HRRR GRIB2 to Zarr converter with better error handling.
Handles time coordinate issues and variable availability.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import zarr
import tqdm
import warnings
from dask.distributed import Client

warnings.filterwarnings("ignore", category=RuntimeWarning)

# HRRR variable definitions with multiple possible names
HRRR_VARS = {
    "REFC": {
        "filters": [{"shortName": "REFC"}, {"parameterName": "Composite reflectivity"}],
        "description": "Composite reflectivity"
    },
    "REFD": {
        "filters": [
            {"shortName": "REFD", "typeOfLevel": "heightAboveGround", "level": 1000},
            {"parameterName": "Reflectivity", "typeOfLevel": "heightAboveGround", "level": 1000}
        ],
        "description": "1km AGL reflectivity"
    },
    "CAPE": {
        "filters": [
            {"shortName": "CAPE", "typeOfLevel": "atmosphereLayer"},
            {"parameterName": "Convective available potential energy", "typeOfLevel": "atmosphereLayer"}
        ],
        "description": "CAPE"
    },
    "CIN": {
        "filters": [
            {"shortName": "CIN", "typeOfLevel": "atmosphereLayer"},
            {"parameterName": "Convective inhibition", "typeOfLevel": "atmosphereLayer"}
        ],
        "description": "CIN"
    },
    "ACPCP": {
        "filters": [{"shortName": "ACPCP"}, {"parameterName": "Convective precipitation"}],
        "description": "Convective precipitation"
    },
    "TMP": {
        "filters": [
            {"shortName": "TMP", "typeOfLevel": "heightAboveGround", "level": 2},
            {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}
        ],
        "description": "2m temperature"
    },
    "DPT": {
        "filters": [
            {"shortName": "DPT", "typeOfLevel": "heightAboveGround", "level": 2},
            {"shortName": "2d", "typeOfLevel": "heightAboveGround", "level": 2}
        ],
        "description": "2m dewpoint"
    },
    "UGRD": {
        "filters": [
            {"shortName": "UGRD", "typeOfLevel": "heightAboveGround", "level": 10},
            {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}
        ],
        "description": "10m U-wind"
    },
    "VGRD": {
        "filters": [
            {"shortName": "VGRD", "typeOfLevel": "heightAboveGround", "level": 10},
            {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10}
        ],
        "description": "10m V-wind"
    }
}

def extract_time_from_filename(fp: Path):
    """Extract datetime from HRRR filename pattern."""
    name = fp.name
    # Try common HRRR patterns
    patterns = [
        r'hrrr\.(\d{8})\.t(\d{2})z',  # hrrr.20250101.t00z
        r'hrrr_(\d{8})_(\d{2})',       # hrrr_20250101_00
        r'(\d{8}).*t(\d{2})z',         # 20250101.t00z
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            date_str = match.group(1)
            hour_str = match.group(2)
            return pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}:00")
    
    return None

def open_hrrr_file(fp: Path, requested_vars=None):
    """Open HRRR GRIB2 file with robust handling."""
    if requested_vars is None:
        requested_vars = list(HRRR_VARS.keys())
    
    datasets = {}
    file_time = extract_time_from_filename(fp)
    
    for var_name in requested_vars:
        var_info = HRRR_VARS.get(var_name, {})
        filters = var_info.get("filters", [])
        
        # Try each filter option
        for filt in filters:
            try:
                ds = xr.open_dataset(fp, engine="cfgrib", backend_kwargs={"filter_by_keys": filt})
                
                # Standardize time coordinate
                if 'valid_time' in ds.coords and 'time' not in ds.coords:
                    ds = ds.rename({'valid_time': 'time'})
                elif 'step' in ds.coords and 'time' not in ds.coords:
                    # For forecast files
                    if file_time is not None:
                        time_val = file_time + pd.Timedelta(hours=int(ds.step.values / 3600000000000))
                        ds = ds.assign_coords(time=time_val)
                    else:
                        ds = ds.rename({'step': 'time'})
                elif 'time' not in ds.coords and file_time is not None:
                    # Use filename time as fallback
                    ds = ds.assign_coords(time=file_time)
                
                # Standardize variable names
                if len(ds.data_vars) == 1:
                    old_name = list(ds.data_vars)[0]
                    ds = ds.rename({old_name: var_name})
                
                datasets[var_name] = ds
                break  # Success, move to next variable
                
            except Exception as e:
                continue
    
    if not datasets:
        return None
    
    # Merge all successful reads
    merged = xr.merge(list(datasets.values()))
    
    # Ensure consistent coordinates
    if 'time' in merged.coords:
        # Expand dimensions if needed
        if 'time' not in merged.dims:
            merged = merged.expand_dims('time')
    
    return merged

def process_file_batch(files, out_dir):
    """Process a batch of files."""
    valid_datasets = []
    
    for fp in files:
        print(f"Processing {fp.name}...")
        ds = open_hrrr_file(fp)
        
        if ds is not None and len(ds.data_vars) > 0:
            # Check for required coordinates
            if all(coord in ds.coords for coord in ['time', 'latitude', 'longitude']):
                valid_datasets.append(ds)
                print(f"  ✓ Loaded {len(ds.data_vars)} variables")
            else:
                print(f"  ✗ Missing required coordinates")
        else:
            print(f"  ✗ No valid data")
    
    return valid_datasets

def main(src_dir: Path, out_dir: Path, chunks: str = "time=1,y=768,x=768"):
    """Main processing function."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GRIB2 files
    files = sorted(src_dir.glob("**/*.grib2"))
    if not files:
        files = sorted(src_dir.glob("**/*.grb2"))
    if not files:
        files = sorted(src_dir.glob("**/*.grib"))
    
    if not files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"Found {len(files)} GRIB2 files")
    
    # Process files
    with Client(n_workers=2, threads_per_worker=2, memory_limit='8GB') as client:
        print(client)
        
        # Process in smaller batches to avoid memory issues
        batch_size = 10
        all_datasets = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
            
            batch_datasets = process_file_batch(batch, out_dir)
            all_datasets.extend(batch_datasets)
        
        if not all_datasets:
            raise ValueError("No valid datasets found! Check your GRIB2 files.")
        
        print(f"\nConcatenating {len(all_datasets)} datasets...")
        
        # Sort by time before concatenating
        all_datasets.sort(key=lambda ds: ds.time.values[0])
        
        # Concatenate all datasets
        full = xr.concat(all_datasets, dim='time')
        
        # Remove duplicate times if any
        _, unique_indices = np.unique(full.time.values, return_index=True)
        full = full.isel(time=unique_indices)
        
        # Sort by time
        full = full.sortby('time')
        
        # Apply chunking
        chunk_dict = {}
        for dim_chunk in chunks.split(','):
            dim, size = dim_chunk.split('=')
            chunk_dict[dim] = int(size)
        
        # Map common dimension names
        if 'y' in chunk_dict and 'y' in full.dims:
            full = full.chunk({'y': chunk_dict['y']})
        elif 'latitude' in full.dims:
            full = full.chunk({'latitude': chunk_dict.get('y', 768)})
            
        if 'x' in chunk_dict and 'x' in full.dims:
            full = full.chunk({'x': chunk_dict['x']})
        elif 'longitude' in full.dims:
            full = full.chunk({'longitude': chunk_dict.get('x', 768)})
            
        if 'time' in chunk_dict:
            full = full.chunk({'time': chunk_dict['time']})
        
        # Write to Zarr
        print("\nWriting to Zarr format...")
        store = zarr.DirectoryStore(out_dir / "hrrr.zarr")
        
        # Use compute=False for dask arrays
        full.to_zarr(store, mode='w', consolidated=True, compute=True)
        
        # Write metadata
        meta = {
            "variables": list(full.data_vars),
            "dimensions": dict(full.dims),
            "shape": dict(full.sizes),
            "chunks": {k: v for k, v in full.chunks.items() if v is not None},
            "time_range": {
                "start": str(full.time.min().values),
                "end": str(full.time.max().values),
                "steps": len(full.time)
            },
            "coordinate_system": {
                "latitude_range": [float(full.latitude.min()), float(full.latitude.max())],
                "longitude_range": [float(full.longitude.min()), float(full.longitude.max())]
            }
        }
        
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n✅ Successfully wrote {len(full.time)} timesteps to Zarr")
        print(f"   Variables: {', '.join(full.data_vars)}")
        print(f"   Time range: {meta['time_range']['start']} to {meta['time_range']['end']}")
        print(f"   Grid size: {full.sizes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 files to Zarr")
    parser.add_argument("--src", required=True, type=Path, help="Source directory with GRIB2 files")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for Zarr")
    parser.add_argument("--chunks", default="time=1,y=768,x=768", help="Chunk specification")
    
    args = parser.parse_args()
    main(args.src, args.out, args.chunks)