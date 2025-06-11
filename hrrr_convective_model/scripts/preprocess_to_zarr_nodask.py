#!/usr/bin/env python
"""
Convert HRRR GRIB2 files to Zarr without Dask (avoids compatibility issues).
Processes files sequentially to avoid memory issues.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import zarr
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Key HRRR variables for convective modeling
VARS = {
    "REFC": {},  # Composite reflectivity
    "REFD": {"typeOfLevel": "heightAboveGround", "level": 1000},  # 1km reflectivity
    "CAPE": {"typeOfLevel": "surface"},  # Surface-based CAPE
    "CIN": {"typeOfLevel": "surface"},   # Surface-based CIN
    "ACPCP": {},  # Accumulated precipitation
    "TMP": {"typeOfLevel": "heightAboveGround", "level": 2},   # 2m temperature
    "DPT": {"typeOfLevel": "heightAboveGround", "level": 2},   # 2m dewpoint
    "UGRD": {"typeOfLevel": "heightAboveGround", "level": 10}, # 10m U-wind
    "VGRD": {"typeOfLevel": "heightAboveGround", "level": 10}, # 10m V-wind
}


def open_single_file(fp: Path):
    """Open one GRIB2 file and extract relevant variables."""
    print(f"  Processing {fp.name}...")
    datasets = []
    
    for var_name, filters in VARS.items():
        try:
            # Try to open with cfgrib
            ds = xr.open_dataset(
                fp, 
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"shortName": var_name, **filters},
                    "indexpath": ""  # Don't create index files
                }
            )
            
            # Standardize coordinate names
            if 'valid_time' in ds.coords and 'time' not in ds.coords:
                ds = ds.rename({'valid_time': 'time'})
            if 'latitude' in ds.coords and 'lat' not in ds.coords:
                ds = ds.rename({'latitude': 'lat'})
            if 'longitude' in ds.coords and 'lon' not in ds.coords:
                ds = ds.rename({'longitude': 'lon'})
            
            # Keep only the variable we want
            var_names = [v for v in ds.data_vars if v not in ['lat', 'lon', 'time']]
            if var_names:
                ds = ds[var_names]
                datasets.append(ds)
                print(f"    ✓ Found {var_name}")
            
        except Exception as e:
            # Try alternative filters for CAPE/CIN
            if var_name in ["CAPE", "CIN"]:
                try:
                    ds = xr.open_dataset(
                        fp,
                        engine="cfgrib", 
                        backend_kwargs={
                            "filter_by_keys": {
                                "shortName": var_name,
                                "typeOfLevel": "atmosphereLayer",
                                "bottomLevel": 255,
                                "topLevel": 0
                            },
                            "indexpath": ""
                        }
                    )
                    if 'valid_time' in ds.coords:
                        ds = ds.rename({'valid_time': 'time'})
                    if 'latitude' in ds.coords:
                        ds = ds.rename({'latitude': 'lat'})
                    if 'longitude' in ds.coords:
                        ds = ds.rename({'longitude': 'lon'})
                    datasets.append(ds)
                    print(f"    ✓ Found {var_name} (atmosphere layer)")
                except:
                    print(f"    ✗ Could not load {var_name}")
            else:
                print(f"    ✗ Could not load {var_name}: {str(e)[:50]}...")
    
    if not datasets:
        return None
    
    # Merge all variables
    merged = xr.merge(datasets)
    
    # Ensure we have required dimensions
    if not all(dim in merged.dims for dim in ['time', 'lat', 'lon']):
        print(f"    ✗ Missing required dimensions")
        return None
    
    return merged


def main(src_dir: Path, out_dir: Path):
    """Main processing function."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GRIB2 files
    files = sorted(src_dir.glob("*.grib2"))
    if not files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"\nFound {len(files)} GRIB2 files\n")
    
    # Process each file
    all_datasets = []
    for fp in files:
        ds = open_single_file(fp)
        if ds is not None:
            all_datasets.append(ds)
        else:
            print(f"  ⚠ Skipping {fp.name} (no valid data)")
    
    if not all_datasets:
        raise ValueError("No valid data extracted from any GRIB2 file!")
    
    print(f"\n✓ Successfully processed {len(all_datasets)} files")
    print("Concatenating datasets...")
    
    # Concatenate along time dimension
    full_ds = xr.concat(all_datasets, dim="time").sortby("time")
    
    # Remove duplicate times if any
    _, unique_indices = np.unique(full_ds.time.values, return_index=True)
    if len(unique_indices) < len(full_ds.time):
        print(f"Removing {len(full_ds.time) - len(unique_indices)} duplicate timestamps")
        full_ds = full_ds.isel(time=unique_indices)
    
    # Apply chunking for efficient storage
    chunk_sizes = {
        'time': 1,
        'lat': min(512, len(full_ds.lat)),
        'lon': min(512, len(full_ds.lon))
    }
    full_ds = full_ds.chunk(chunk_sizes)
    
    # Write to Zarr
    zarr_path = out_dir / "hrrr.zarr"
    print(f"\nWriting to {zarr_path}...")
    
    # Use safe encoding
    encoding = {}
    for var in full_ds.data_vars:
        encoding[var] = {
            'compressor': zarr.Blosc(cname='zstd', clevel=3),
            'dtype': 'float32'
        }
    
    full_ds.to_zarr(
        zarr_path,
        mode='w',
        consolidated=True,
        encoding=encoding
    )
    
    # Write metadata
    meta = {
        "variables": list(full_ds.data_vars),
        "shape": dict(full_ds.sizes),
        "time_range": {
            "start": str(full_ds.time.min().values),
            "end": str(full_ds.time.max().values), 
            "steps": len(full_ds.time)
        },
        "spatial_resolution": {
            "lat_min": float(full_ds.lat.min()),
            "lat_max": float(full_ds.lat.max()),
            "lon_min": float(full_ds.lon.min()),
            "lon_max": float(full_ds.lon.max()),
            "n_lat": len(full_ds.lat),
            "n_lon": len(full_ds.lon)
        }
    }
    
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Success! Wrote {zarr_path}")
    print(f"   Variables: {', '.join(full_ds.data_vars)}")
    print(f"   Shape: time={len(full_ds.time)}, lat={len(full_ds.lat)}, lon={len(full_ds.lon)}")
    print(f"   Time range: {meta['time_range']['start']} to {meta['time_range']['end']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 to Zarr")
    parser.add_argument("--src", required=True, type=Path, help="Source directory with GRIB2 files")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for Zarr")
    args = parser.parse_args()
    
    main(args.src, args.out)