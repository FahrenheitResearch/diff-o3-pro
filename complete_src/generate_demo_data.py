#!/usr/bin/env python3
"""
Generate Demo HRRR Training Data
Creates realistic synthetic NetCDF datasets that mimic HRRR structure for demonstration
"""

import sys
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def create_realistic_hrrr_data(var_name: str, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    """Create realistic synthetic data for HRRR variables"""
    ny, nx = lat_grid.shape
    
    # Create base spatial patterns
    lat_center = np.mean(lat_grid)
    lon_center = np.mean(lon_grid)
    
    # Distance from center for spatial patterns
    lat_dist = (lat_grid - lat_center) / 10.0
    lon_dist = (lon_grid - lon_center) / 10.0
    distance = np.sqrt(lat_dist**2 + lon_dist**2)
    
    # Add realistic noise
    noise = np.random.randn(ny, nx) * 0.3
    
    # Generate realistic data based on variable type
    if 'temperature' in var_name.lower() or 't2m' in var_name.lower():
        # Temperature: decrease with latitude, seasonal variation
        base = 288.0 - lat_dist * 5.0  # Kelvin, cooler northward
        variation = 15.0 * np.sin(distance) + noise * 5.0
        data = base + variation
        
    elif 'wind' in var_name.lower() or any(x in var_name.lower() for x in ['u10', 'v10', 'wspd']):
        # Wind speed: more variable, generally positive
        base = 8.0 + distance * 2.0
        variation = 10.0 * np.abs(np.sin(distance * 2)) + noise * 3.0
        data = np.maximum(0, base + variation)
        
    elif 'pressure' in var_name.lower():
        # Pressure: smooth gradients, realistic range
        base = 1013.25 - distance * 5.0  # hPa
        variation = 20.0 * np.sin(distance * 0.5) + noise * 2.0
        data = base + variation
        
    elif 'cape' in var_name.lower():
        # CAPE: patchy, mostly positive with some zeros
        base = np.maximum(0, 500 + distance * 200)
        variation = 1500 * np.maximum(0, np.sin(distance * 3)) + noise * 100
        data = np.maximum(0, base + variation)
        
    elif 'cin' in var_name.lower():
        # CIN: mostly negative
        base = -50 - distance * 20
        variation = -30 * np.abs(np.sin(distance * 2)) + noise * 10
        data = np.minimum(0, base + variation)
        
    elif 'shear' in var_name.lower():
        # Wind shear: moderate values, spatially coherent
        base = 10.0 + distance * 5.0
        variation = 15.0 * np.sin(distance * 1.5) + noise * 2.0
        data = np.maximum(0, base + variation)
        
    elif 'helicity' in var_name.lower() or 'srh' in var_name.lower():
        # Storm-relative helicity: can be positive or negative
        base = distance * 30.0
        variation = 100.0 * np.sin(distance * 2) + noise * 20.0
        data = base + variation
        
    elif 'humidity' in var_name.lower() or 'rh' in var_name.lower():
        # Humidity: 0-100%
        base = 60.0 - distance * 10.0
        variation = 30.0 * np.sin(distance * 2) + noise * 5.0
        data = np.clip(base + variation, 0, 100)
        
    elif 'cloud' in var_name.lower():
        # Cloud cover: 0-100%
        base = 40.0 + distance * 20.0
        variation = 40.0 * np.sin(distance * 3) + noise * 10.0
        data = np.clip(base + variation, 0, 100)
        
    elif 'visibility' in var_name.lower():
        # Visibility: mostly high with some low areas
        base = 20000 - distance * 2000
        variation = -5000 * np.maximum(0, np.sin(distance * 4)) + noise * 1000
        data = np.maximum(1000, base + variation)
        
    else:
        # Generic meteorological data
        base = 10.0 + distance * 2.0
        variation = 5.0 * np.sin(distance * 2) + noise * 2.0
        data = base + variation
    
    return data


def generate_demo_training_data(
    cycle: str = "2025061002",
    forecast_hours: List[int] = [0, 1, 2, 3],
    output_dir: Path = Path("./demo_hrrr_data"),
    num_variables: int = 15
) -> Dict[str, any]:
    """Generate demo training data with realistic structure"""
    
    print(f"🚀 Generating Demo HRRR Training Data")
    print(f"  Cycle: {cycle}")
    print(f"  Forecast hours: {forecast_hours}")
    print(f"  Output directory: {output_dir}")
    print(f"  Variables: {num_variables}")
    
    # Initialize pipeline to get variable configurations
    pipeline = HRRRTrainingPipeline()
    
    # Select representative variables
    all_variables = pipeline.select_training_variables(
        include_categories=['severe', 'instability', 'surface', 'atmospheric'],
        include_derived=True,
        max_variables=num_variables
    )
    
    # Create realistic HRRR grid (downsampled for demo)
    # Real HRRR is ~3km resolution, we'll use ~25km for demo
    lat_1d = np.linspace(21.0, 47.8, 120)  # CONUS latitude range
    lon_1d = np.linspace(-122.7, -60.9, 180)  # CONUS longitude range
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    
    # Store 1D coordinates for xarray
    lat_coords = lat_1d
    lon_coords = lon_1d
    
    print(f"📊 Grid size: {lat_grid.shape} (~25km resolution)")
    
    # Generate data for each forecast hour
    successful_datasets = 0
    processing_times = []
    
    for fh in forecast_hours:
        print(f"\n📅 Generating F{fh:02d} data...")
        start_time = datetime.now()
        
        try:
            # Create synthetic data for each variable
            fields_data = {}
            
            for var_name, var_config in all_variables.items():
                print(f"  Creating {var_name}...")
                
                # Generate realistic data
                data_values = create_realistic_hrrr_data(var_name, lat_grid, lon_grid)
                
                # Add some realistic missing data (5% NaN)
                mask = np.random.random(data_values.shape) < 0.05
                data_values[mask] = np.nan
                
                # Create xarray DataArray
                data_array = xr.DataArray(
                    data_values,
                    coords={
                        'latitude': ('y', lat_coords),
                        'longitude': ('x', lon_coords)
                    },
                    dims=['y', 'x'],
                    name=var_name,
                    attrs={
                        'long_name': var_config.get('title', var_name),
                        'units': var_config.get('units', 'dimensionless'),
                        'standard_name': var_name,
                        'grid_mapping': 'crs'
                    }
                )
                
                # Validate data quality
                if pipeline.validate_data_quality(data_array, var_name):
                    fields_data[var_name] = data_array
                else:
                    print(f"    ⚠️ Skipping {var_name} due to quality issues")
            
            if not fields_data:
                print(f"  ❌ No valid fields for F{fh:02d}")
                continue
            
            # Create training dataset
            dataset = pipeline.create_training_dataset(
                fields_data,
                cycle,
                fh,
                metadata={
                    'demo_data': 'True',
                    'synthetic': 'True',
                    'grid_resolution': '25km',
                    'note': 'Realistic synthetic data for demonstration purposes'
                }
            )
            
            # Save dataset
            output_path = output_dir / f"cycle_{cycle}" / f"forecast_hour_F{fh:02d}.nc"
            pipeline.save_training_dataset(dataset, output_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_times.append(processing_time)
            successful_datasets += 1
            
            print(f"  ✅ F{fh:02d} completed in {processing_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error generating F{fh:02d}: {e}")
    
    # Generate summary
    summary = {
        'total_datasets': len(forecast_hours),
        'successful_datasets': successful_datasets,
        'failed_datasets': len(forecast_hours) - successful_datasets,
        'success_rate': successful_datasets / len(forecast_hours) if forecast_hours else 0,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0,
        'total_processing_time': sum(processing_times),
        'output_directory': str(output_dir),
        'variables_processed': list(all_variables.keys()),
        'num_variables': len(all_variables),
        'cycle': cycle,
        'forecast_hours': forecast_hours,
        'grid_shape': f"{lat_grid.shape[0]}x{lat_grid.shape[1]}",
        'demo_data': True,
        'generated_at': datetime.now().isoformat()
    }
    
    # Save summary
    summary_file = output_dir / 'generation_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print results
    print(f"\n" + "="*60)
    print(f"📊 DEMO DATA GENERATION SUMMARY")
    print(f"="*60)
    print(f"✅ Successful: {successful_datasets}/{len(forecast_hours)} ({summary['success_rate']*100:.1f}%)")
    print(f"⏱️  Average time per dataset: {summary['avg_processing_time']:.1f}s")
    print(f"🎯 Variables per dataset: {summary['num_variables']}")
    print(f"📏 Grid size: {summary['grid_shape']} (~25km resolution)")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Summary saved to: {summary_file}")
    
    return summary


def inspect_demo_data(output_dir: Path):
    """Inspect the generated demo data"""
    print(f"\n🔍 Inspecting Generated Demo Data")
    print(f"="*40)
    
    nc_files = list(output_dir.rglob('*.nc'))
    
    if not nc_files:
        print("❌ No NetCDF files found")
        return
    
    # Inspect first file
    sample_file = nc_files[0]
    print(f"📁 Sample file: {sample_file}")
    
    try:
        ds = xr.open_dataset(sample_file)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Variables: {len(ds.data_vars)}")
        print(f"   Dimensions: {dict(ds.dims)}")
        print(f"   File size: {sample_file.stat().st_size / (1024*1024):.1f} MB")
        
        print(f"\n🗂️  Sample Variables:")
        for i, var_name in enumerate(list(ds.data_vars.keys())[:5]):
            var = ds[var_name]
            print(f"   {var_name}:")
            print(f"     Range: [{float(np.nanmin(var.values)):.2f}, {float(np.nanmax(var.values)):.2f}]")
            print(f"     Mean: {float(np.nanmean(var.values)):.6f}")
            print(f"     Std: {float(np.nanstd(var.values)):.6f}")
            
            if 'normalization_mean' in var.attrs:
                orig_mean = var.attrs['normalization_mean']
                orig_std = var.attrs['normalization_std']
                print(f"     Original: mean={orig_mean:.3f}, std={orig_std:.3f}")
        
        ds.close()
        
        print(f"\n📂 All generated files:")
        for f in sorted(nc_files):
            size_mb = f.stat().st_size / (1024*1024)
            print(f"   {f.name}: {size_mb:.1f} MB")
            
    except Exception as e:
        print(f"❌ Error inspecting data: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate demo HRRR training data')
    parser.add_argument('--cycle', default='2025061002', help='HRRR cycle (YYYYMMDDHH)')
    parser.add_argument('--forecast-hours', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='Forecast hours to generate')
    parser.add_argument('--output-dir', type=Path, default='./demo_hrrr_data',
                       help='Output directory')
    parser.add_argument('--num-variables', type=int, default=15,
                       help='Number of variables to include')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect generated data after creation')
    
    args = parser.parse_args()
    
    # Generate demo data
    summary = generate_demo_training_data(
        cycle=args.cycle,
        forecast_hours=args.forecast_hours,
        output_dir=args.output_dir,
        num_variables=args.num_variables
    )
    
    # Inspect if requested
    if args.inspect:
        inspect_demo_data(args.output_dir)
    
    print(f"\n🎉 Demo data generation complete!")
    print(f"Use these NetCDF files as examples for your diffusion model training.")


if __name__ == '__main__':
    main()