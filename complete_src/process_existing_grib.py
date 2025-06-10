#!/usr/bin/env python3
"""
Process Existing GRIB File to NetCDF
Use the already downloaded 390MB HRRR GRIB file
"""

import sys
import time
import numpy as np
import xarray as xr
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def process_existing_grib():
    """Process the existing downloaded GRIB file"""
    
    print("🚀 Processing Existing Real HRRR GRIB File")
    print("="*50)
    
    # Use the already downloaded GRIB file
    grib_file = Path('./real_hrrr_native_3km/grib_cache/hrrr.t00z.wrfprsf00.grib2')
    wrfsfc_file = Path('./real_hrrr_native_3km/grib_cache/hrrr.t00z.wrfsfcf00.grib2')
    
    if not grib_file.exists():
        print(f"❌ GRIB file not found: {grib_file}")
        return None
    
    file_size_mb = grib_file.stat().st_size / (1024 * 1024)
    print(f"✅ Found GRIB file: {grib_file.name} ({file_size_mb:.1f} MB)")
    
    if wrfsfc_file.exists():
        sfc_size_mb = wrfsfc_file.stat().st_size / (1024 * 1024)
        print(f"✅ Found surface file: {wrfsfc_file.name} ({sfc_size_mb:.1f} MB)")
    else:
        wrfsfc_file = None
        print("⚠️ No surface file found (continuing with pressure levels only)")
    
    # Initialize pipeline
    pipeline = HRRRTrainingPipeline()
    
    # Select core variables that we know work with HRRR
    core_variables = {
        # Surface meteorological fields (basic and reliable)
        't2m': {
            'title': '2m Temperature',
            'units': 'K',
            'category': 'surface',
            'var': 't2m',
            'access': {'paramId': 167, 'typeOfLevel': 'heightAboveGround', 'level': 2}
        },
        'd2m': {
            'title': '2m Dewpoint',
            'units': 'K',
            'category': 'surface',
            'var': 'd2m',
            'access': {'paramId': 168, 'typeOfLevel': 'heightAboveGround', 'level': 2}
        },
        'u10': {
            'title': '10m U-Wind',
            'units': 'm/s',
            'category': 'surface',
            'var': 'u10',
            'access': {'paramId': 165, 'typeOfLevel': 'heightAboveGround', 'level': 10}
        },
        'v10': {
            'title': '10m V-Wind',
            'units': 'm/s',
            'category': 'surface',
            'var': 'v10',
            'access': {'paramId': 166, 'typeOfLevel': 'heightAboveGround', 'level': 10}
        },
        'sp': {
            'title': 'Surface Pressure',
            'units': 'Pa',
            'category': 'surface',
            'var': 'sp',
            'access': {'paramId': 134, 'typeOfLevel': 'surface'}
        },
        # Atmospheric fields that loaded successfully
        'tcc': {
            'title': 'Total Cloud Cover',
            'units': '%',
            'category': 'atmospheric',
            'var': 'tcc',
            'access': {'paramId': 164, 'typeOfLevel': 'atmosphere'}
        },
        'vis': {
            'title': 'Visibility',
            'units': 'm',
            'category': 'atmospheric',
            'var': 'vis',
            'access': {'paramId': 20, 'typeOfLevel': 'surface'}
        }
    }
    
    print(f"📊 Processing {len(core_variables)} core variables:")
    for var_name, config in core_variables.items():
        print(f"   {var_name}: {config['title']} ({config['category']})")
    
    # Load field data
    print(f"\n🔄 Loading fields at native 3km resolution...")
    start_time = time.time()
    
    fields_data = {}
    
    for field_name, field_config in core_variables.items():
        print(f"  Loading: {field_name}")
        
        try:
            data = pipeline.load_field_data_for_training(
                field_name, field_config, grib_file, wrfsfc_file
            )
            
            if data is not None and pipeline.validate_data_quality(data, field_name):
                fields_data[field_name] = data
                print(f"    ✅ {data.shape} - range [{float(data.min()):.2f}, {float(data.max()):.2f}]")
            else:
                print(f"    ❌ Failed")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    if not fields_data:
        print("❌ No valid fields loaded")
        return None
    
    print(f"\n📊 Successfully loaded {len(fields_data)} fields")
    
    # Verify grid size
    sample_field = next(iter(fields_data.values()))
    grid_shape = sample_field.shape
    total_points = grid_shape[0] * grid_shape[1]
    
    print(f"✅ Native HRRR grid: {grid_shape[0]}×{grid_shape[1]} = {total_points:,} points")
    
    # Create training dataset
    print(f"\n🔧 Creating training dataset...")
    
    dataset = pipeline.create_training_dataset(
        fields_data,
        cycle="2025060900",
        forecast_hour=0,
        metadata={
            'real_hrrr_data': True,
            'native_resolution': True,
            'grid_points': total_points,
            'grib_file_size_mb': file_size_mb,
            'processing_time_seconds': time.time() - start_time,
            'note': 'Real HRRR data at native 3km resolution'
        }
    )
    
    # Save NetCDF file
    output_file = Path('./real_hrrr_native_3km/hrrr_real_native_2025060900_F00.nc')
    print(f"\n💾 Saving to: {output_file}")
    
    pipeline.save_training_dataset(dataset, output_file)
    
    # Verify output
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Real HRRR training data created!")
        print(f"📁 File: {output_file}")
        print(f"📊 Size: {file_size_mb:.1f} MB")
        print(f"🌐 Grid: {total_points:,} points at 3km resolution")
        print(f"📈 Variables: {len(fields_data)}")
        print(f"⏱️  Processing time: {time.time() - start_time:.1f} seconds")
        
        # Show variable details
        print(f"\n📋 Variables included:")
        for var_name, var_data in fields_data.items():
            config = core_variables[var_name]
            mean_val = float(np.nanmean(var_data.values))
            std_val = float(np.nanstd(var_data.values))
            print(f"   {var_name}: {config['title']} (mean={mean_val:.2f}, std={std_val:.2f})")
        
        return output_file
    else:
        print("❌ Output file was not created")
        return None


def inspect_real_output(nc_file: Path):
    """Inspect the real HRRR NetCDF output"""
    
    print(f"\n🔍 INSPECTING REAL HRRR OUTPUT")
    print("="*40)
    
    if not nc_file.exists():
        print(f"❌ File not found: {nc_file}")
        return
    
    # Load and inspect
    ds = xr.open_dataset(nc_file)
    
    file_size_mb = nc_file.stat().st_size / (1024 * 1024)
    total_points = ds.sizes['lat'] * ds.sizes['lon']
    
    print(f"📁 File: {nc_file.name}")
    print(f"📊 Size: {file_size_mb:.1f} MB")
    print(f"🌐 Grid: {ds.sizes['lat']}×{ds.sizes['lon']} = {total_points:,} points")
    print(f"📈 Variables: {len(ds.data_vars)}")
    
    print(f"\n🏷️  Global Attributes:")
    for attr, value in ds.attrs.items():
        print(f"   {attr}: {value}")
    
    print(f"\n📊 Variable Statistics:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        values = var.values[~np.isnan(var.values)]
        
        if len(values) > 0:
            print(f"\n   {var_name}:")
            print(f"     Title: {var.attrs.get('long_name', 'N/A')}")
            print(f"     Units: {var.attrs.get('units', 'N/A')}")
            print(f"     Shape: {var.shape}")
            print(f"     Range: [{values.min():.3f}, {values.max():.3f}]")
            print(f"     Normalized: mean={np.mean(values):.6f}, std={np.std(values):.6f}")
            
            # Original statistics
            if 'normalization_mean' in var.attrs:
                orig_mean = var.attrs['normalization_mean']
                orig_std = var.attrs['normalization_std']
                print(f"     Original: mean={orig_mean:.3f}, std={orig_std:.3f}")
    
    ds.close()
    
    print(f"\n✅ This is REAL HRRR data at native 3km resolution!")
    print(f"   Use this NetCDF file for your diffusion model training.")


if __name__ == '__main__':
    # Process the existing GRIB file
    output_file = process_existing_grib()
    
    if output_file:
        # Inspect the output
        inspect_real_output(output_file)
        
        print(f"\n🎯 REAL HRRR TRAINING DATA READY!")
        print(f"File: {output_file}")
        print(f"This is genuine HRRR data at native 3km resolution.")
    else:
        print(f"\n❌ Failed to process GRIB file")