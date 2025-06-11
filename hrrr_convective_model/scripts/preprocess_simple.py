#!/usr/bin/env python
"""
Simple GRIB2 to Zarr converter without Dask.
Processes files sequentially to avoid memory issues.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import zarr
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings("ignore")

# Core variables to extract
VARS = {
    "REFC": {},  # Composite reflectivity
    "CAPE": {"typeOfLevel": "atmosphereLayer", "bottomLevel": 255, "topLevel": 0},
    "CIN": {"typeOfLevel": "atmosphereLayer", "bottomLevel": 255, "topLevel": 0},
    "TMP": {"typeOfLevel": "heightAboveGround", "level": 2},
    "DPT": {"typeOfLevel": "heightAboveGround", "level": 2},
    "UGRD": {"typeOfLevel": "heightAboveGround", "level": 10},
    "VGRD": {"typeOfLevel": "heightAboveGround", "level": 10},
}

def open_grib_file(filepath):
    """Open a GRIB2 file and extract specified variables."""
    all_vars = {}
    coords_ref = None
    
    for var_name, filters in VARS.items():
        try:
            # Try to open with specific filters
            filter_keys = {"shortName": var_name}
            filter_keys.update(filters)
            
            ds = xr.open_dataset(
                filepath,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_keys}
            )
            
            # Get the actual variable name in the dataset
            actual_var_names = list(ds.data_vars)
            if not actual_var_names:
                print(f"  ✗ No data vars found for {var_name}")
                continue
                
            # HRRR often uses different names, get the first data variable
            actual_var = actual_var_names[0]
            
            # Store coordinate reference from first successful load
            if coords_ref is None:
                coords_ref = ds.coords
                # Ensure we have a time coordinate
                if 'valid_time' in coords_ref and 'time' not in coords_ref:
                    coords_ref = coords_ref.rename({'valid_time': 'time'})
            
            # Extract just the data variable with consistent coordinates
            data_array = ds[actual_var]
            
            # Rename coordinates to be consistent
            coord_mapping = {}
            if 'valid_time' in data_array.coords:
                coord_mapping['valid_time'] = 'time'
            if 'y' in data_array.dims and 'latitude' not in data_array.dims:
                coord_mapping['y'] = 'latitude'  
            if 'x' in data_array.dims and 'longitude' not in data_array.dims:
                coord_mapping['x'] = 'longitude'
                
            if coord_mapping:
                data_array = data_array.rename(coord_mapping)
            
            # Store with standardized name
            all_vars[var_name] = data_array
            print(f"  ✓ Loaded {var_name} (actual: {actual_var})")
            
        except Exception as e:
            print(f"  ✗ Could not load {var_name}: {str(e)[:100]}")
            continue
    
    if not all_vars:
        return None
    
    # Create dataset from variables
    merged = xr.Dataset(all_vars)
    
    # Add time dimension if needed
    if 'time' not in merged.coords:
        # Extract time from the first variable that has it
        for var in merged.data_vars:
            if 'time' in merged[var].coords:
                merged = merged.assign_coords(time=merged[var].time)
                break
        else:
            # If still no time, try to get from filename
            import re
            match = re.search(r't(\d{2})z', filepath.name)
            if match:
                hour = int(match.group(1))
                date_match = re.search(r'(\d{8})', filepath.name)
                if date_match:
                    from datetime import datetime
                    date_str = date_match.group(1)
                    time_val = datetime.strptime(f"{date_str}{hour:02d}", "%Y%m%d%H")
                    merged = merged.assign_coords(time=np.datetime64(time_val))
    
    # Ensure time is a dimension not just coordinate
    if 'time' in merged.coords and 'time' not in merged.dims:
        merged = merged.expand_dims('time')
    
    print(f"  → Dataset has {len(merged.data_vars)} variables: {list(merged.data_vars)}")
    
    return merged

def process_sequential(grib_files, output_path):
    """Process GRIB files sequentially without Dask."""
    all_datasets = []
    
    print(f"\nProcessing {len(grib_files)} GRIB2 files sequentially...\n")
    
    for i, grib_file in enumerate(grib_files):
        print(f"[{i+1}/{len(grib_files)}] Processing {grib_file.name}...")
        
        try:
            ds = open_grib_file(grib_file)
            if ds is not None and len(ds.data_vars) > 0:
                all_datasets.append(ds)
                print(f"  → Successfully loaded with {len(ds.data_vars)} variables\n")
            else:
                print(f"  → No valid data extracted\n")
        except Exception as e:
            print(f"  → Error: {str(e)[:100]}\n")
            continue
    
    if not all_datasets:
        raise ValueError("No valid data extracted from any GRIB2 file!")
    
    print(f"Concatenating {len(all_datasets)} datasets along time dimension...")
    
    # Concatenate along time dimension
    combined = xr.concat(all_datasets, dim='time')
    
    # Sort by time
    if 'time' in combined.dims:
        combined = combined.sortby('time')
    
    # Add some metadata
    combined.attrs['source'] = 'HRRR GRIB2 files'
    combined.attrs['processed_date'] = datetime.now().isoformat()
    
    # Chunk for efficient storage (but don't use dask)
    chunk_sizes = {
        'time': 1,
        'latitude': 512,
        'longitude': 512
    }
    
    # Apply chunking for zarr storage
    actual_chunks = {}
    for dim in combined.dims:
        if dim in chunk_sizes:
            actual_chunks[dim] = chunk_sizes[dim]
        elif dim == 'y':
            actual_chunks[dim] = chunk_sizes.get('latitude', 512)
        elif dim == 'x':
            actual_chunks[dim] = chunk_sizes.get('longitude', 512)
    
    # Write to Zarr
    print(f"\nWriting to Zarr at {output_path}...")
    
    # Create zarr store
    store = zarr.DirectoryStore(str(output_path))
    
    # Write with encoding
    encoding = {}
    for var in combined.data_vars:
        encoding[var] = {
            'chunks': tuple(actual_chunks.get(d, -1) for d in combined[var].dims),
            'compressor': zarr.Blosc(cname='zstd', clevel=3)
        }
    
    combined.to_zarr(store, mode='w', encoding=encoding, consolidated=True)
    
    return combined

def main(src_dir, out_dir):
    """Main processing function."""
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all GRIB2 files
    grib_files = sorted(src_path.glob("*.grib2"))
    
    if not grib_files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"Found {len(grib_files)} GRIB2 files")
    
    # Process files
    zarr_path = out_path / "hrrr.zarr"
    dataset = process_sequential(grib_files, zarr_path)
    
    # Write metadata
    metadata = {
        "variables": list(dataset.data_vars),
        "dimensions": dict(dataset.sizes),
        "time_range": {
            "start": str(dataset.time.min().values),
            "end": str(dataset.time.max().values),
            "steps": len(dataset.time)
        },
        "source_files": len(grib_files)
    }
    
    with open(out_path / "manifest.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Successfully converted GRIB2 to Zarr!")
    print(f"   Output: {zarr_path}")
    print(f"   Variables: {', '.join(dataset.data_vars)}")
    print(f"   Time steps: {len(dataset.time)}")
    print(f"   Dimensions: {dict(dataset.sizes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 to Zarr (no Dask)")
    parser.add_argument("--src", required=True, help="Source directory with GRIB2 files")
    parser.add_argument("--out", required=True, help="Output directory for Zarr")
    args = parser.parse_args()
    
    main(args.src, args.out)