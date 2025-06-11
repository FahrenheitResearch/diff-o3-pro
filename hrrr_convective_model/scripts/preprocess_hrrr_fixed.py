#!/usr/bin/env python
"""
HRRR GRIB2 to Zarr converter with correct variable mappings.
Based on actual HRRR data structure discovered through debugging.
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

# HRRR variable mappings discovered from actual files
# Format: "desired_name": (shortName, filters, expected_var_name)
HRRR_VARS = {
    "REFC": ("refc", {}, "refc"),  # Composite reflectivity
    "T2M": ("2t", {}, "t2m"),      # 2m temperature
    "D2M": ("2d", {}, "d2m"),      # 2m dewpoint
    "U10": ("10u", {}, "u10"),     # 10m u-wind
    "V10": ("10v", {}, "v10"),     # 10m v-wind
    "CAPE": ("cape", {"typeOfLevel": "surface"}, "cape"),  # Surface CAPE
    "CIN": ("cin", {"typeOfLevel": "surface"}, "cin"),     # Surface CIN
}

# Additional variables to try with different filters
HRRR_VARS_ALTERNATE = {
    "CAPE_SFC": ("cape", {"typeOfLevel": "surface"}, "cape"),
    "CAPE_ML": ("cape", {"typeOfLevel": "atmosphereLayer", "bottomLevel": 255, "topLevel": 0}, "cape"),
    "CIN_SFC": ("cin", {"typeOfLevel": "surface"}, "cin"),
    "CIN_ML": ("cin", {"typeOfLevel": "atmosphereLayer", "bottomLevel": 255, "topLevel": 0}, "cin"),
    "APCP": ("tp", {}, "tp"),  # Total precipitation
}

def extract_time_from_filename(filename):
    """Extract datetime from HRRR filename."""
    # Pattern: hrrr.YYYYMMDD.tHHz.wrfprsf00.grib2
    match = re.search(r'hrrr\.(\d{8})\.t(\d{2})z', filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
        return np.datetime64(dt)
    return None

def open_hrrr_grib(filepath):
    """Open HRRR GRIB2 file with correct variable mappings."""
    extracted_vars = {}
    file_time = extract_time_from_filename(filepath.name)
    
    print(f"\nProcessing {filepath.name}...")
    if file_time:
        print(f"  File time: {file_time}")
    
    # Try primary variables
    for desired_name, (short_name, filters, expected_var) in HRRR_VARS.items():
        try:
            filter_keys = {"shortName": short_name}
            filter_keys.update(filters)
            
            ds = xr.open_dataset(
                filepath,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_keys}
            )
            
            if expected_var in ds.data_vars:
                var_data = ds[expected_var]
                
                # Drop all non-essential coordinates to avoid conflicts
                coords_to_drop = []
                for coord in var_data.coords:
                    if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                        coords_to_drop.append(coord)
                
                if coords_to_drop:
                    var_data = var_data.drop_vars(coords_to_drop)
                
                # Ensure it's a DataArray
                if hasattr(var_data, 'to_array'):
                    var_data = var_data.to_array().squeeze()
                    if 'variable' in var_data.dims:
                        var_data = var_data.drop_vars('variable')
                
                extracted_vars[desired_name] = var_data
                print(f"  ✓ {desired_name}: shape {var_data.shape}")
            else:
                # Try to get the first data variable if expected name doesn't match
                if len(ds.data_vars) > 0:
                    first_var = list(ds.data_vars)[0]
                    var_data = ds[first_var]
                    
                    # Drop all non-essential coordinates
                    coords_to_drop = []
                    for coord in var_data.coords:
                        if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                            coords_to_drop.append(coord)
                    
                    if coords_to_drop:
                        var_data = var_data.drop_vars(coords_to_drop)
                    
                    extracted_vars[desired_name] = var_data
                    print(f"  ✓ {desired_name} (as {first_var}): shape {var_data.shape}")
                
        except Exception as e:
            # Try alternate filters
            for alt_name, (alt_short, alt_filters, alt_expected) in HRRR_VARS_ALTERNATE.items():
                if alt_name.startswith(desired_name):
                    try:
                        filter_keys = {"shortName": alt_short}
                        filter_keys.update(alt_filters)
                        
                        ds = xr.open_dataset(
                            filepath,
                            engine="cfgrib",
                            backend_kwargs={"filter_by_keys": filter_keys}
                        )
                        
                        if len(ds.data_vars) > 0:
                            first_var = list(ds.data_vars)[0]
                            var_data = ds[first_var]
                            
                            # Drop all non-essential coordinates
                            coords_to_drop = []
                            for coord in var_data.coords:
                                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                                    coords_to_drop.append(coord)
                            
                            if coords_to_drop:
                                var_data = var_data.drop_vars(coords_to_drop)
                            
                            extracted_vars[desired_name] = var_data
                            print(f"  ✓ {desired_name} (alternate): shape {var_data.shape}")
                            break
                    except:
                        continue
            
            if desired_name not in extracted_vars:
                print(f"  ✗ {desired_name}: {str(e)[:50]}...")
    
    if not extracted_vars:
        return None
    
    # Create dataset with consistent structure
    ds_out = xr.Dataset(extracted_vars)
    
    # Add time coordinate
    if file_time is not None:
        ds_out = ds_out.assign_coords(time=file_time)
    
    # Ensure time is a dimension
    if 'time' in ds_out.coords and 'time' not in ds_out.dims:
        ds_out = ds_out.expand_dims('time')
    
    # Get coordinate info from first variable
    first_var = list(ds_out.data_vars)[0]
    if 'latitude' in ds_out[first_var].coords:
        ds_out = ds_out.assign_coords({
            'latitude': ds_out[first_var].latitude,
            'longitude': ds_out[first_var].longitude
        })
    
    print(f"  → Extracted {len(ds_out.data_vars)} variables")
    
    return ds_out

def process_to_zarr(grib_files, output_path):
    """Process HRRR GRIB files to Zarr."""
    all_datasets = []
    
    for grib_file in grib_files:
        try:
            ds = open_hrrr_grib(grib_file)
            if ds is not None and len(ds.data_vars) > 0:
                all_datasets.append(ds)
        except Exception as e:
            print(f"  Error processing {grib_file.name}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No valid data extracted from GRIB files!")
    
    print(f"\nConcatenating {len(all_datasets)} datasets...")
    
    # Concatenate along time
    combined = xr.concat(all_datasets, dim='time')
    
    # Sort by time
    combined = combined.sortby('time')
    
    # Add metadata
    combined.attrs.update({
        'source': 'HRRR GRIB2 files',
        'processing_date': datetime.now().isoformat(),
        'grid': '3km CONUS'
    })
    
    # Define chunking
    chunks = {
        'time': 1,
        'y': 256,
        'x': 256
    }
    
    # Create encoding
    encoding = {}
    for var in combined.data_vars:
        var_chunks = tuple(chunks.get(dim, -1) for dim in combined[var].dims)
        encoding[var] = {
            'chunks': var_chunks,
            'compressor': zarr.Blosc(cname='zstd', clevel=3)
        }
    
    # Write to Zarr
    print(f"\nWriting to {output_path}...")
    store = zarr.DirectoryStore(str(output_path))
    combined.to_zarr(store, mode='w', encoding=encoding, consolidated=True)
    
    return combined

def main(src_dir, out_dir):
    """Main processing function."""
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find GRIB files
    grib_files = sorted(src_path.glob("*.grib2"))
    
    if not grib_files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    print(f"Found {len(grib_files)} GRIB2 files")
    
    # Process to Zarr
    zarr_path = out_path / "hrrr.zarr"
    dataset = process_to_zarr(grib_files, zarr_path)
    
    # Write metadata
    metadata = {
        "variables": list(dataset.data_vars),
        "dimensions": dict(dataset.sizes),
        "coordinates": list(dataset.coords),
        "time_range": {
            "start": str(dataset.time.min().values),
            "end": str(dataset.time.max().values),
            "steps": len(dataset.time)
        },
        "grid_info": {
            "projection": "Lambert Conformal",
            "resolution": "3km",
            "domain": "CONUS"
        }
    }
    
    with open(out_path / "manifest.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ Successfully converted HRRR GRIB2 to Zarr!")
    print(f"   Output: {zarr_path}")
    print(f"   Variables: {', '.join(dataset.data_vars)}")
    print(f"   Time steps: {len(dataset.time)}")
    print(f"   Grid dimensions: y={dataset.sizes['y']}, x={dataset.sizes['x']}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 to Zarr")
    parser.add_argument("--src", required=True, help="Source directory with GRIB2 files")
    parser.add_argument("--out", required=True, help="Output directory for Zarr")
    args = parser.parse_args()
    
    main(args.src, args.out)