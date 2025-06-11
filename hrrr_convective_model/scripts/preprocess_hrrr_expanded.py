#!/usr/bin/env python
"""
Expanded HRRR GRIB2 to Zarr converter for DEF implementation.
Extracts full atmospheric state at multiple pressure levels + forcings.
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

# think-harder: Expanding variable set to match DEF paper requirements
# Table 1: Atmospheric state variables at multiple pressure levels
# Table 2: Forcing variables (solar radiation, time features)

# Pressure levels to extract (hPa) - standard atmospheric levels
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Expanded HRRR variable mappings for DEF
HRRR_VARS_3D = {
    # Geopotential height at pressure levels
    "Z": {
        "shortName": "gh",
        "typeOfLevel": "isobaricInhPa", 
        "expected_var": "gh"
    },
    # Temperature at pressure levels
    "T": {
        "shortName": "t",
        "typeOfLevel": "isobaricInhPa",
        "expected_var": "t"
    },
    # Specific humidity at pressure levels
    "Q": {
        "shortName": "q",
        "typeOfLevel": "isobaricInhPa",
        "expected_var": "q"
    },
    # U-wind component at pressure levels
    "U": {
        "shortName": "u",
        "typeOfLevel": "isobaricInhPa",
        "expected_var": "u"
    },
    # V-wind component at pressure levels
    "V": {
        "shortName": "v",
        "typeOfLevel": "isobaricInhPa",
        "expected_var": "v"
    }
}

# Surface variables
HRRR_VARS_SURFACE = {
    "SP": ("sp", {}, "sp"),                                   # Surface pressure
    "T2M": ("2t", {}, "t2m"),                                # 2m temperature
    "D2M": ("2d", {}, "d2m"),                                # 2m dewpoint
    "U10": ("10u", {}, "u10"),                               # 10m u-wind
    "V10": ("10v", {}, "v10"),                               # 10m v-wind
    "CAPE": ("cape", {"typeOfLevel": "surface"}, "cape"),   # Surface CAPE
    "CIN": ("cin", {"typeOfLevel": "surface"}, "cin"),      # Surface CIN
    "REFC": ("refc", {}, "refc"),                           # Composite reflectivity
}

# Forcing variables
HRRR_VARS_FORCING = {
    "DSWRF": ("dswrf", {}, "dswrf"),  # Downward shortwave radiation flux
    "DLWRF": ("dlwrf", {}, "dlwrf"),  # Downward longwave radiation flux
    "PWAT": ("pwat", {}, "pwat"),     # Precipitable water
}

# Try alternate names if primary extraction fails
HRRR_VARS_ALTERNATE = {
    "SP": [("pres", {"typeOfLevel": "surface"}, "pres"), 
           ("PRES", {"typeOfLevel": "surface"}, "PRES")],
    "DSWRF": [("DSWRF", {}, "DSWRF"), 
              ("SWDOWN", {}, "SWDOWN")],
    "Z": [("HGT", {"typeOfLevel": "isobaricInhPa"}, "HGT"),
          ("hgt", {"typeOfLevel": "isobaricInhPa"}, "hgt")],
    "T": [("TMP", {"typeOfLevel": "isobaricInhPa"}, "TMP"),
          ("tmp", {"typeOfLevel": "isobaricInhPa"}, "tmp")],
    "Q": [("SPFH", {"typeOfLevel": "isobaricInhPa"}, "SPFH"),
          ("spfh", {"typeOfLevel": "isobaricInhPa"}, "spfh")],
    "U": [("UGRD", {"typeOfLevel": "isobaricInhPa"}, "UGRD"),
          ("ugrd", {"typeOfLevel": "isobaricInhPa"}, "ugrd")],
    "V": [("VGRD", {"typeOfLevel": "isobaricInhPa"}, "VGRD"),
          ("vgrd", {"typeOfLevel": "isobaricInhPa"}, "vgrd")]
}

def extract_time_from_filename(filename):
    """Extract datetime from HRRR filename."""
    match = re.search(r'hrrr\.(\d{8})\.t(\d{2})z', filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
        return np.datetime64(dt)
    return None

def extract_3d_variable(filepath, var_name, var_info):
    """Extract a 3D variable at all pressure levels."""
    extracted_levels = {}
    
    for level in PRESSURE_LEVELS:
        try:
            filter_keys = {
                "shortName": var_info["shortName"],
                "typeOfLevel": var_info["typeOfLevel"],
                "level": level
            }
            
            ds = xr.open_dataset(
                filepath,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_keys}
            )
            
            if var_info["expected_var"] in ds.data_vars:
                var_data = ds[var_info["expected_var"]]
            elif len(ds.data_vars) > 0:
                var_data = ds[list(ds.data_vars)[0]]
            else:
                continue
                
            # Clean up coordinates
            coords_to_drop = []
            for coord in var_data.coords:
                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                    coords_to_drop.append(coord)
            if coords_to_drop:
                var_data = var_data.drop_vars(coords_to_drop)
                
            extracted_levels[f"{var_name}_{level}"] = var_data
            
        except Exception as e:
            # Try alternate names
            if var_name in HRRR_VARS_ALTERNATE:
                for alt_short, alt_filters, alt_expected in HRRR_VARS_ALTERNATE[var_name]:
                    try:
                        filter_keys = {
                            "shortName": alt_short,
                            "level": level
                        }
                        filter_keys.update(alt_filters)
                        
                        ds = xr.open_dataset(
                            filepath,
                            engine="cfgrib",
                            backend_kwargs={"filter_by_keys": filter_keys}
                        )
                        
                        if len(ds.data_vars) > 0:
                            var_data = ds[list(ds.data_vars)[0]]
                            coords_to_drop = []
                            for coord in var_data.coords:
                                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                                    coords_to_drop.append(coord)
                            if coords_to_drop:
                                var_data = var_data.drop_vars(coords_to_drop)
                            extracted_levels[f"{var_name}_{level}"] = var_data
                            break
                    except:
                        continue
    
    return extracted_levels

def open_hrrr_grib_expanded(filepath):
    """Open HRRR GRIB2 file with expanded variable set."""
    extracted_vars = {}
    file_time = extract_time_from_filename(filepath.name)
    
    print(f"\nProcessing {filepath.name}...")
    if file_time:
        print(f"  File time: {file_time}")
    
    # Extract 3D variables at pressure levels
    print("  Extracting 3D variables...")
    for var_name, var_info in HRRR_VARS_3D.items():
        levels_data = extract_3d_variable(filepath, var_name, var_info)
        extracted_vars.update(levels_data)
        if levels_data:
            print(f"    ✓ {var_name}: extracted {len(levels_data)} levels")
        else:
            print(f"    ✗ {var_name}: no levels extracted")
    
    # Extract surface variables
    print("  Extracting surface variables...")
    for desired_name, (short_name, filters, expected_var) in HRRR_VARS_SURFACE.items():
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
            elif len(ds.data_vars) > 0:
                var_data = ds[list(ds.data_vars)[0]]
            else:
                raise ValueError(f"No data found for {desired_name}")
                
            # Clean coordinates
            coords_to_drop = []
            for coord in var_data.coords:
                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                    coords_to_drop.append(coord)
            if coords_to_drop:
                var_data = var_data.drop_vars(coords_to_drop)
                
            extracted_vars[desired_name] = var_data
            print(f"    ✓ {desired_name}")
            
        except Exception as e:
            # Try alternates
            if desired_name in HRRR_VARS_ALTERNATE:
                for alt_short, alt_filters, alt_expected in HRRR_VARS_ALTERNATE[desired_name]:
                    try:
                        filter_keys = {"shortName": alt_short}
                        filter_keys.update(alt_filters)
                        ds = xr.open_dataset(
                            filepath,
                            engine="cfgrib",
                            backend_kwargs={"filter_by_keys": filter_keys}
                        )
                        if len(ds.data_vars) > 0:
                            var_data = ds[list(ds.data_vars)[0]]
                            coords_to_drop = []
                            for coord in var_data.coords:
                                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                                    coords_to_drop.append(coord)
                            if coords_to_drop:
                                var_data = var_data.drop_vars(coords_to_drop)
                            extracted_vars[desired_name] = var_data
                            print(f"    ✓ {desired_name} (alternate)")
                            break
                    except:
                        continue
            if desired_name not in extracted_vars:
                print(f"    ✗ {desired_name}")
    
    # Extract forcing variables
    print("  Extracting forcing variables...")
    for desired_name, (short_name, filters, expected_var) in HRRR_VARS_FORCING.items():
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
            elif len(ds.data_vars) > 0:
                var_data = ds[list(ds.data_vars)[0]]
            else:
                continue
                
            coords_to_drop = []
            for coord in var_data.coords:
                if coord not in ['latitude', 'longitude', 'y', 'x'] and coord not in var_data.dims:
                    coords_to_drop.append(coord)
            if coords_to_drop:
                var_data = var_data.drop_vars(coords_to_drop)
                
            extracted_vars[desired_name] = var_data
            print(f"    ✓ {desired_name}")
            
        except:
            print(f"    ✗ {desired_name}")
    
    if not extracted_vars:
        return None
    
    # Create dataset
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
    
    print(f"  → Total extracted variables: {len(ds_out.data_vars)}")
    
    return ds_out

def process_to_zarr_expanded(grib_files, output_path):
    """Process HRRR GRIB files to Zarr with expanded variables."""
    all_datasets = []
    
    for i, grib_file in enumerate(grib_files):
        print(f"\n[{i+1}/{len(grib_files)}]", end='')
        try:
            ds = open_hrrr_grib_expanded(grib_file)
            if ds is not None and len(ds.data_vars) > 0:
                all_datasets.append(ds)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No valid data extracted from GRIB files!")
    
    print(f"\n\nConcatenating {len(all_datasets)} datasets...")
    
    # Find common variables across all files
    common_vars = set(all_datasets[0].data_vars)
    for ds in all_datasets[1:]:
        common_vars &= set(ds.data_vars)
    
    print(f"Common variables across all files: {len(common_vars)}")
    print(f"Keeping only common variables to ensure consistency...")
    
    # Filter datasets to common variables
    filtered_datasets = []
    for ds in all_datasets:
        filtered_datasets.append(ds[list(common_vars)])
    
    # Concatenate along time
    combined = xr.concat(filtered_datasets, dim='time')
    combined = combined.sortby('time')
    
    # Add metadata
    combined.attrs.update({
        'source': 'HRRR GRIB2 files - Extended DEF variables',
        'processing_date': datetime.now().isoformat(),
        'grid': '3km CONUS',
        'pressure_levels': str(PRESSURE_LEVELS),
        'num_3d_vars': len([v for v in common_vars if any(str(p) in v for p in PRESSURE_LEVELS)]),
        'num_surface_vars': len([v for v in common_vars if not any(str(p) in v for p in PRESSURE_LEVELS)])
    })
    
    # Define chunking - adjust for larger dataset
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
            'compressor': zarr.Blosc(cname='zstd', clevel=3),
            'dtype': 'float32'
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
    print(f"Extracting expanded variable set for DEF implementation...")
    print(f"Target pressure levels: {PRESSURE_LEVELS}")
    
    # Process to Zarr
    zarr_path = out_path / "hrrr_expanded.zarr"
    dataset = process_to_zarr_expanded(grib_files, zarr_path)
    
    # Write metadata
    all_vars = list(dataset.data_vars)
    var_3d = [v for v in all_vars if any(str(p) in v for p in PRESSURE_LEVELS)]
    var_surface = [v for v in all_vars if v in HRRR_VARS_SURFACE]
    var_forcing = [v for v in all_vars if v in HRRR_VARS_FORCING]
    
    metadata = {
        "variables": {
            "total": len(all_vars),
            "3d": var_3d,
            "surface": var_surface,
            "forcing": var_forcing,
            "all": all_vars
        },
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
        },
        "pressure_levels": PRESSURE_LEVELS
    }
    
    with open(out_path / "manifest_expanded.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Successfully converted HRRR GRIB2 to Zarr (Expanded)!")
    print(f"   Output: {zarr_path}")
    print(f"   Total variables: {len(all_vars)}")
    print(f"   3D variables: {len(var_3d)} ({len(var_3d)//len(PRESSURE_LEVELS)} vars × {len(PRESSURE_LEVELS)} levels)")
    print(f"   Surface variables: {len(var_surface)}")
    print(f"   Forcing variables: {len(var_forcing)}")
    print(f"   Time steps: {len(dataset.time)}")
    print(f"   Grid: y={dataset.sizes['y']}, x={dataset.sizes['x']}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HRRR GRIB2 to Zarr with expanded DEF variables")
    parser.add_argument("--src", required=True, help="Source directory with GRIB2 files")
    parser.add_argument("--out", required=True, help="Output directory for Zarr")
    args = parser.parse_args()
    
    main(args.src, args.out)