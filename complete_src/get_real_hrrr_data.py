#!/usr/bin/env python3
"""
Download and Process REAL HRRR Data at Native 3km Resolution
No synthetic data - only actual HRRR GRIB files processed to NetCDF
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import xarray as xr

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def download_real_hrrr_grib(cycle: str, forecast_hour: int, 
                           output_dir: Path, file_type: str = 'wrfprs') -> Optional[Path]:
    """Download real HRRR GRIB file with aggressive retry strategy"""
    
    print(f"🌪️  Downloading REAL HRRR data: {cycle} F{forecast_hour:02d}")
    
    cycle_dt = datetime.strptime(cycle, '%Y%m%d%H')
    date_str = cycle_dt.strftime('%Y%m%d')
    
    # Build filename
    filename = f'hrrr.t{cycle[-2:]}z.{file_type}f{forecast_hour:02d}.grib2'
    output_path = output_dir / filename
    
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # Real HRRR files should be > 50MB
            print(f"✅ Real HRRR file exists: {filename} ({file_size_mb:.1f} MB)")
            return output_path
        else:
            print(f"⚠️ File too small ({file_size_mb:.1f} MB), re-downloading...")
            output_path.unlink()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try multiple data sources with longer retention
    sources = [
        # NOMADS (most recent)
        f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}',
        # AWS S3 (historical)
        f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/{filename}',
        # University of Utah Pando (backup)
        f'https://pando-rgw01.chpc.utah.edu/hrrr/hrrr.{date_str}/conus/{filename}',
        # Alternative AWS paths
        f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/{filename}',
    ]
    
    for i, url in enumerate(sources):
        try:
            source_name = ["NOMADS", "AWS S3", "Utah Pando", "AWS Alt"][i]
            print(f"  🔄 Trying {source_name}: {url}")
            
            # Set longer timeout for large files (HRRR files are 100-500 MB)
            import socket
            socket.setdefaulttimeout(1200)  # 20 minutes
            
            # Download with progress indication
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if percent % 10 == 0 and block_num * block_size > 0:
                        mb_downloaded = (block_num * block_size) / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"    📥 {percent}% - {mb_downloaded:.1f}/{total_mb:.1f} MB")
            
            urllib.request.urlretrieve(url, output_path, progress_hook)
            
            # Verify download
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 50:  # Real HRRR files should be substantial
                    print(f"✅ Successfully downloaded from {source_name}: {filename} ({file_size_mb:.1f} MB)")
                    return output_path
                else:
                    print(f"❌ Downloaded file too small ({file_size_mb:.1f} MB) - likely corrupted")
                    output_path.unlink()
                    
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  ❌ {source_name}: File not found (404)")
            else:
                print(f"  ❌ {source_name}: HTTP error {e.code}")
        except Exception as e:
            print(f"  ❌ {source_name}: Download failed - {e}")
        
        # Small delay between attempts
        time.sleep(1)
    
    print(f"❌ Failed to download {filename} from all sources")
    return None


def validate_real_hrrr_grib(grib_file: Path) -> bool:
    """Validate that we have a real HRRR GRIB file"""
    
    print(f"🔍 Validating GRIB file: {grib_file.name}")
    
    try:
        import cfgrib
        
        # Check file size first
        file_size_mb = grib_file.stat().st_size / (1024 * 1024)
        if file_size_mb < 50:
            print(f"❌ File too small: {file_size_mb:.1f} MB (real HRRR files are 100-500 MB)")
            return False
        
        print(f"✅ File size OK: {file_size_mb:.1f} MB")
        
        # Try to open with cfgrib using surface level (most common for training)
        try:
            # Test with surface level first (most HRRR training data is from here)
            ds = cfgrib.open_dataset(grib_file, 
                                   filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'},
                                   backend_kwargs={'indexpath': ''})
            
            # Check dimensions - real HRRR should have ~1059x1799 grid points
            total_points = 1
            spatial_dims = []
            for dim, size in ds.dims.items():
                if dim in ['y', 'x', 'latitude', 'longitude']:
                    total_points *= size
                    spatial_dims.append(f"{dim}={size}")
            
            print(f"✅ Spatial dimensions: {', '.join(spatial_dims)}")
            print(f"✅ Total grid points: {total_points:,}")
            
            if total_points < 1000000:  # Should be ~1.9M for real HRRR
                print(f"⚠️ Grid seems small for HRRR (expected ~1.9M points)")
                # Don't fail - some HRRR products have different grids
            
            # Check for typical HRRR variables
            vars_found = list(ds.data_vars.keys())
            print(f"✅ Variables found: {len(vars_found)}")
            print(f"  Sample variables: {vars_found[:10]}")
            
            ds.close()
            
            # Also test heightAboveGround level
            try:
                ds2 = cfgrib.open_dataset(grib_file,
                                        filter_by_keys={'typeOfLevel': 'heightAboveGround'},
                                        backend_kwargs={'indexpath': ''})
                vars_found2 = list(ds2.data_vars.keys())
                print(f"✅ HeightAboveGround variables: {len(vars_found2)}")
                print(f"  Sample variables: {vars_found2[:5]}")
                ds2.close()
            except:
                print("⚠️ Could not load heightAboveGround level (not critical)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to open GRIB with cfgrib: {e}")
            return False
            
    except Exception as e:
        print(f"❌ GRIB validation failed: {e}")
        return False


def find_available_hrrr_cycle() -> Optional[str]:
    """Find the most recent available HRRR cycle by testing downloads"""
    
    print("🔍 Searching for available HRRR cycles...")
    
    # Try cycles from the last 7 days, every 6 hours
    now = datetime.utcnow()
    
    for days_back in range(1, 8):  # 1-7 days ago
        for hour in [0, 6, 12, 18]:  # Standard HRRR cycles
            test_dt = now - timedelta(days=days_back)
            test_dt = test_dt.replace(hour=hour, minute=0, second=0, microsecond=0)
            test_cycle = test_dt.strftime('%Y%m%d%H')
            
            print(f"  Testing cycle: {test_cycle}")
            
            # Quick test - just check if URL exists
            date_str = test_dt.strftime('%Y%m%d')
            test_url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/hrrr.t{hour:02d}z.wrfprsf00.grib2'
            
            try:
                import urllib.request
                req = urllib.request.Request(test_url, method='HEAD')
                response = urllib.request.urlopen(req, timeout=10)
                if response.status == 200:
                    print(f"✅ Found available cycle: {test_cycle}")
                    return test_cycle
            except:
                continue
    
    print("❌ No available HRRR cycles found in the last 7 days")
    return None


def process_real_hrrr_data(cycle: str, forecast_hour: int, 
                          output_dir: Path, max_variables: int = 20) -> Optional[Path]:
    """Download and process real HRRR data at native resolution"""
    
    print(f"\n🚀 PROCESSING REAL HRRR DATA")
    print(f"="*50)
    print(f"Cycle: {cycle}")
    print(f"Forecast Hour: F{forecast_hour:02d}")
    print(f"Output Directory: {output_dir}")
    print(f"Max Variables: {max_variables}")
    
    # Step 1: Download real GRIB files
    grib_cache_dir = output_dir / 'grib_cache'
    
    grib_file = download_real_hrrr_grib(cycle, forecast_hour, grib_cache_dir, 'wrfprs')
    if not grib_file:
        print("❌ Failed to download HRRR GRIB file")
        return None
    
    # Step 2: Validate the GRIB file
    if not validate_real_hrrr_grib(grib_file):
        print("❌ GRIB file validation failed")
        return None
    
    # Step 3: Download auxiliary file if needed (for smoke, updraft helicity)
    wrfsfc_file = download_real_hrrr_grib(cycle, forecast_hour, grib_cache_dir, 'wrfsfc')
    if wrfsfc_file:
        print(f"✅ Also downloaded auxiliary file: {wrfsfc_file.name}")
    
    # Step 4: Process with training pipeline at NATIVE resolution
    print(f"\n📊 Processing at NATIVE 3km resolution...")
    
    pipeline = HRRRTrainingPipeline()
    
    # Select variables for comprehensive weather data
    variables = pipeline.select_training_variables(
        include_categories=['severe', 'instability', 'surface', 'atmospheric'],
        exclude_categories=['smoke'],  # Skip smoke for now (memory intensive)
        include_derived=True,
        max_variables=max_variables
    )
    
    print(f"Selected {len(variables)} variables:")
    for i, (var_name, var_config) in enumerate(variables.items(), 1):
        category = var_config.get('category', 'unknown')
        title = var_config.get('title', var_name)
        print(f"  {i:2d}. {var_name}: {title} ({category})")
    
    # Step 5: Load all field data at native resolution
    print(f"\n🔄 Loading field data at native resolution...")
    start_time = time.time()
    
    fields_data = {}
    failed_fields = []
    
    for field_name, field_config in variables.items():
        print(f"  Loading: {field_name}")
        
        try:
            data = pipeline.load_field_data_for_training(
                field_name, field_config, grib_file, wrfsfc_file
            )
            
            if data is not None and pipeline.validate_data_quality(data, field_name):
                fields_data[field_name] = data
                print(f"    ✅ Loaded: {data.shape} - range [{float(data.min()):.2f}, {float(data.max()):.2f}]")
            else:
                failed_fields.append(field_name)
                print(f"    ❌ Failed quality validation")
                
        except Exception as e:
            failed_fields.append(field_name)
            print(f"    ❌ Error: {e}")
    
    if not fields_data:
        print("❌ No valid fields loaded")
        return None
    
    print(f"\n📊 Successfully loaded {len(fields_data)}/{len(variables)} fields")
    if failed_fields:
        print(f"Failed fields: {', '.join(failed_fields)}")
    
    # Step 6: Create training dataset
    print(f"\n🔧 Creating training dataset...")
    
    # Get actual grid size
    sample_field = next(iter(fields_data.values()))
    grid_shape = sample_field.shape
    total_points = grid_shape[0] * grid_shape[1]
    
    print(f"Grid size: {grid_shape[0]}×{grid_shape[1]} = {total_points:,} points")
    
    dataset = pipeline.create_training_dataset(
        fields_data,
        cycle,
        forecast_hour,
        metadata={
            'real_hrrr_data': True,
            'grib_file': str(grib_file),
            'native_resolution': True,
            'grid_points': total_points,
            'processing_time_seconds': time.time() - start_time
        }
    )
    
    # Step 7: Save NetCDF file
    output_file = output_dir / f"hrrr_native_{cycle}_F{forecast_hour:02d}.nc"
    pipeline.save_training_dataset(dataset, output_file)
    
    # Step 8: Verify the output
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Output file: {output_file}")
        print(f"✅ File size: {file_size_mb:.1f} MB")
        print(f"✅ Variables: {len(fields_data)}")
        print(f"✅ Grid points: {total_points:,}")
        print(f"✅ Processing time: {time.time() - start_time:.1f} seconds")
        
        return output_file
    else:
        print("❌ Output file was not created")
        return None


def main():
    """Main function to get real HRRR training data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and process REAL HRRR data at native resolution')
    parser.add_argument('--cycle', help='HRRR cycle (YYYYMMDDHH) - will auto-detect if not provided')
    parser.add_argument('--forecast-hour', type=int, default=0, help='Forecast hour (default: 0)')
    parser.add_argument('--output-dir', type=Path, default='./real_hrrr_native', help='Output directory')
    parser.add_argument('--max-variables', type=int, default=25, help='Maximum variables to process')
    
    args = parser.parse_args()
    
    # Find available cycle if not specified
    if not args.cycle:
        args.cycle = find_available_hrrr_cycle()
        if not args.cycle:
            print("❌ Could not find any available HRRR cycles")
            return 1
    
    # Process the data
    output_file = process_real_hrrr_data(
        cycle=args.cycle,
        forecast_hour=args.forecast_hour,
        output_dir=args.output_dir,
        max_variables=args.max_variables
    )
    
    if output_file:
        print(f"\n🎯 REAL HRRR TRAINING DATA READY!")
        print(f"File: {output_file}")
        print(f"Use this for your diffusion model training.")
        return 0
    else:
        print(f"\n❌ Failed to generate real HRRR training data")
        return 1


if __name__ == '__main__':
    sys.exit(main())