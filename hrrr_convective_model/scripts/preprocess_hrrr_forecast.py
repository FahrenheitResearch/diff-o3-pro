#!/usr/bin/env python
"""
HRRR GRIB2 to Zarr converter with forecast hour support.
Processes multiple forecast hours (F00-F18) to create training sequences.
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

# HRRR variable mappings
HRRR_VARS = {
    "REFC": ("refc", {}, "refc"),  # Composite reflectivity
    "T2M": ("2t", {}, "t2m"),      # 2m temperature
    "D2M": ("2d", {}, "d2m"),      # 2m dewpoint
    "U10": ("10u", {}, "u10"),     # 10m u-wind
    "V10": ("10v", {}, "v10"),     # 10m v-wind
    "CAPE": ("cape", {"typeOfLevel": "surface"}, "cape"),  # Surface CAPE
    "CIN": ("cin", {"typeOfLevel": "surface"}, "cin"),     # Surface CIN
}

def extract_time_info_from_filename(filename):
    """Extract datetime and forecast hour from HRRR filename."""
    # Pattern: hrrr.YYYYMMDD.tHHz.wrfprsfFF.grib2
    match = re.search(r'hrrr\.(\d{8})\.t(\d{2})z\.wrfprsf(\d{2})\.grib2', filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        forecast_hour = int(match.group(3))
        dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
        return np.datetime64(dt), forecast_hour
    return None, None

def open_hrrr_grib(filepath):
    """Open HRRR GRIB2 file with correct variable mappings."""
    extracted_vars = {}
    init_time, forecast_hour = extract_time_info_from_filename(filepath.name)
    
    print(f"\nProcessing {filepath.name}...")
    if init_time is not None:
        print(f"  Initialization time: {init_time}, Forecast hour: F{forecast_hour:02d}")
    
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
                
                # Drop any singleton dimensions
                var_data = var_data.squeeze()
                
                # Ensure we have 2D data
                if var_data.ndim == 2:
                    extracted_vars[desired_name] = var_data
                    print(f"  ✓ Extracted {desired_name}: shape {var_data.shape}")
        except Exception as e:
            print(f"  ✗ Could not extract {desired_name}: {str(e)[:100]}")
    
    if not extracted_vars:
        print(f"  WARNING: No variables extracted from {filepath.name}")
        return None
    
    # Combine all variables
    combined = xr.Dataset(extracted_vars)
    
    # Add forecast metadata
    combined.attrs['init_time'] = str(init_time) if init_time else "unknown"
    combined.attrs['forecast_hour'] = forecast_hour if forecast_hour is not None else -1
    
    return combined, init_time, forecast_hour


def process_forecast_cycle(cycle_files, zarr_store, time_idx):
    """Process a complete forecast cycle (F00-F18)."""
    # Sort files by forecast hour
    sorted_files = sorted(cycle_files, key=lambda f: int(re.search(r'wrfprsf(\d{2})', f.name).group(1)))
    
    print(f"\nProcessing forecast cycle with {len(sorted_files)} files")
    
    # Track which forecast hours we've processed
    processed_hours = {}
    
    for filepath in sorted_files:
        result = open_hrrr_grib(filepath)
        if result is None:
            continue
            
        combined, init_time, forecast_hour = result
        
        # Store data for this forecast hour
        processed_hours[forecast_hour] = combined
    
    # Create forecast sequences for training
    # Each sequence is: state(t) -> state(t+1)
    sequences = []
    for fh in sorted(processed_hours.keys())[:-1]:  # All but last hour
        if fh + 1 in processed_hours:
            sequences.append({
                'input_fh': fh,
                'target_fh': fh + 1,
                'input_data': processed_hours[fh],
                'target_data': processed_hours[fh + 1]
            })
    
    print(f"Created {len(sequences)} training sequences")
    
    # Write sequences to zarr
    for seq in sequences:
        # Each sequence gets its own time index
        for var_name in HRRR_VARS.keys():
            if var_name in seq['input_data'].data_vars:
                # Store input state
                input_data = seq['input_data'][var_name].values
                zarr_store[var_name][time_idx, :, :] = input_data
                
                # Store metadata
                zarr_store.attrs[f'time_{time_idx}_init'] = seq['input_data'].attrs['init_time']
                zarr_store.attrs[f'time_{time_idx}_fh'] = seq['input_fh']
                zarr_store.attrs[f'time_{time_idx}_type'] = 'input'
        
        time_idx += 1
        
        # Store target state
        for var_name in HRRR_VARS.keys():
            if var_name in seq['target_data'].data_vars:
                target_data = seq['target_data'][var_name].values
                zarr_store[var_name][time_idx, :, :] = target_data
                
                zarr_store.attrs[f'time_{time_idx}_init'] = seq['target_data'].attrs['init_time']
                zarr_store.attrs[f'time_{time_idx}_fh'] = seq['target_fh']
                zarr_store.attrs[f'time_{time_idx}_type'] = 'target'
        
        time_idx += 1
    
    return time_idx


def main(args):
    """Convert HRRR GRIB2 files to Zarr format with forecast sequences."""
    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all GRIB2 files
    grib_files = sorted(src_path.glob("*.grib2"))
    print(f"Found {len(grib_files)} GRIB2 files")
    
    if not grib_files:
        print("No GRIB2 files found!")
        return
    
    # Group files by initialization cycle
    cycles = {}
    for f in grib_files:
        match = re.search(r'hrrr\.(\d{8})\.t(\d{2})z', f.name)
        if match:
            cycle_key = f"{match.group(1)}_{match.group(2)}"
            if cycle_key not in cycles:
                cycles[cycle_key] = []
            cycles[cycle_key].append(f)
    
    print(f"Found {len(cycles)} forecast cycles")
    
    # Get dimensions from first file
    first_result = open_hrrr_grib(grib_files[0])
    if first_result is None:
        print("Could not read first file!")
        return
    
    first_ds, _, _ = first_result
    first_var = list(first_ds.data_vars.values())[0]
    ny, nx = first_var.shape
    
    # Calculate total time steps needed
    # Each cycle with N forecast hours produces N-1 sequences
    # Each sequence uses 2 time steps (input and target)
    total_sequences = sum(len(files) - 1 for files in cycles.values() if len(files) > 1)
    total_timesteps = total_sequences * 2
    
    print(f"\nDataset dimensions:")
    print(f"  Spatial: {ny} x {nx}")
    print(f"  Forecast cycles: {len(cycles)}")
    print(f"  Training sequences: {total_sequences}")
    print(f"  Total timesteps: {total_timesteps}")
    
    # Create Zarr store
    zarr_path = out_path / "hrrr.zarr"
    store = zarr.open_group(str(zarr_path), mode='w')
    
    # Create arrays for each variable
    for var_name in HRRR_VARS.keys():
        store.create_dataset(
            var_name,
            shape=(total_timesteps, ny, nx),
            chunks=(1, ny, nx),
            dtype='float32'
        )
    
    # Store coordinates
    if 'latitude' in first_ds.coords:
        lat = first_ds.latitude.values
        lon = first_ds.longitude.values
        store.create_dataset('latitude', data=lat, dtype='float32')
        store.create_dataset('longitude', data=lon, dtype='float32')
    
    # Create time array
    time_array = np.empty(total_timesteps, dtype='datetime64[ns]')
    store.create_dataset('time', shape=(total_timesteps,), dtype='int64')
    
    # Process each forecast cycle
    time_idx = 0
    for cycle_idx, (cycle_key, cycle_files) in enumerate(sorted(cycles.items())):
        print(f"\nProcessing cycle {cycle_idx + 1}/{len(cycles)}: {cycle_key}")
        if len(cycle_files) > 1:  # Need at least 2 files for a sequence
            time_idx = process_forecast_cycle(cycle_files, store, time_idx)
    
    # Store metadata
    store.attrs['variables'] = list(HRRR_VARS.keys())
    store.attrs['total_sequences'] = total_sequences
    store.attrs['created'] = datetime.now().isoformat()
    
    # Create manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "variables": list(HRRR_VARS.keys()),
        "dimensions": {"time": total_timesteps, "y": ny, "x": nx},
        "forecast_cycles": len(cycles),
        "training_sequences": total_sequences
    }
    
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output: {zarr_path}")
    print(f"  Training sequences: {total_sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 files to Zarr with forecast sequences")
    parser.add_argument("--src", type=str, required=True, help="Source directory with GRIB2 files")
    parser.add_argument("--out", type=str, required=True, help="Output directory for Zarr store")
    
    args = parser.parse_args()
    main(args)