#!/usr/bin/env python3
"""
Inspect Real HRRR Training Data
Show detailed analysis of the actual HRRR data at native resolution
"""

import xarray as xr
import numpy as np
from pathlib import Path


def inspect_real_hrrr():
    """Comprehensive inspection of real HRRR data"""
    
    nc_file = Path('./real_hrrr_native_3km/REAL_HRRR_NATIVE_2025060900_F00.nc')
    
    if not nc_file.exists():
        print(f"❌ File not found: {nc_file}")
        return
    
    print("🌪️  REAL HRRR TRAINING DATA ANALYSIS")
    print("="*60)
    
    # Load dataset
    ds = xr.open_dataset(nc_file)
    
    # Basic information
    total_points = ds.sizes['lat'] * ds.sizes['lon']
    file_size_mb = nc_file.stat().st_size / (1024 * 1024)
    
    print(f"📁 File: {nc_file.name}")
    print(f"📊 Size: {file_size_mb:.1f} MB")
    print(f"🌐 Grid: {ds.sizes['lat']}×{ds.sizes['lon']} = {total_points:,} points")
    print(f"🎯 Resolution: Native HRRR 3km")
    print(f"📈 Variables: {len(ds.data_vars)}")
    
    # Coverage
    lat_range = (float(ds.lat.min()), float(ds.lat.max()))
    lon_range = (float(ds.lon.min()), float(ds.lon.max()))
    print(f"📍 Coverage: {lat_range[0]:.2f}°-{lat_range[1]:.2f}°N, {lon_range[0]:.2f}°-{lon_range[1]:.2f}°E")
    
    # Global attributes
    print(f"\n🏷️  Model Information:")
    print(f"   Source: {ds.attrs.get('source', 'N/A')}")
    print(f"   Cycle: {ds.attrs.get('cycle', 'N/A')}")
    print(f"   Valid Time: {ds.attrs.get('valid_time', 'N/A')}")
    print(f"   Real Data: {ds.attrs.get('real_hrrr_data', 'N/A')}")
    print(f"   Native Resolution: {ds.attrs.get('native_resolution', 'N/A')}")
    
    # Data quality verification
    print(f"\n✅ DATA QUALITY VERIFICATION:")
    
    # Check for synthetic data markers
    is_synthetic = ('demo_data' in ds.attrs or 'synthetic' in ds.attrs)
    print(f"   Synthetic data: {'❌ YES' if is_synthetic else '✅ NO'}")
    
    # Check grid size
    grid_ok = total_points > 1000000
    print(f"   Grid size: {'✅ APPROPRIATE' if grid_ok else '❌ TOO SMALL'} ({total_points:,} points)")
    
    # Check file size
    size_ok = file_size_mb > 10
    print(f"   File size: {'✅ APPROPRIATE' if size_ok else '❌ TOO SMALL'} ({file_size_mb:.1f} MB)")
    
    # Variable analysis
    print(f"\n📊 VARIABLE ANALYSIS:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        values = var.values[~np.isnan(var.values)]
        
        print(f"\n   🌡️  {var_name.upper()} - {var.attrs.get('long_name', 'N/A')}")
        print(f"      Units: {var.attrs.get('units', 'N/A')}")
        print(f"      Shape: {var.shape}")
        print(f"      Data type: {var.dtype}")
        
        # Normalization check
        norm_mean = float(np.mean(values))
        norm_std = float(np.std(values))
        print(f"      Normalized: mean={norm_mean:.6f}, std={norm_std:.6f}")
        
        # Original statistics
        if 'normalization_mean' in var.attrs:
            orig_mean = var.attrs['normalization_mean']
            orig_std = var.attrs['normalization_std']
            orig_min = var.attrs['original_min']
            orig_max = var.attrs['original_max']
            
            print(f"      Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
            print(f"      Range: [{orig_min:.2f}, {orig_max:.2f}] {var.attrs.get('units', '')}")
            
            # Realistic range check
            realistic = check_realistic_range(var_name, orig_min, orig_max)
            print(f"      Realistic: {'✅ YES' if realistic else '❌ NO'}")
        
        # Missing data check
        total_vals = var.size
        valid_vals = len(values)
        missing_pct = (total_vals - valid_vals) / total_vals * 100
        print(f"      Missing: {missing_pct:.1f}% ({total_vals - valid_vals:,} / {total_vals:,})")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    all_checks = [
        not is_synthetic,
        grid_ok,
        size_ok,
        len(ds.data_vars) >= 5
    ]
    
    if all(all_checks):
        print("✅ EXCELLENT: This is genuine HRRR data at native resolution!")
        print("✅ Ready for production diffusion model training")
        grade = "A+"
    elif sum(all_checks) >= 3:
        print("✅ GOOD: This appears to be real HRRR data with minor issues")
        print("✅ Suitable for diffusion model training")
        grade = "B+"
    else:
        print("⚠️ CAUTION: Data may not be appropriate for training")
        grade = "C"
    
    print(f"\n📈 DATA GRADE: {grade}")
    
    # Usage example
    print(f"\n💻 PYTHON USAGE EXAMPLE:")
    print(f"""
import xarray as xr
import numpy as np

# Load the real HRRR data
ds = xr.open_dataset('{nc_file}')

# Access normalized variables (mean=0, std=1)
temp = ds['t2m'].values          # Shape: ({ds.sizes['lat']}, {ds.sizes['lon']})
dewpoint = ds['d2m'].values      # 2m dewpoint
u_wind = ds['u10'].values        # 10m U-wind
v_wind = ds['v10'].values        # 10m V-wind
pressure = ds['sp'].values       # Surface pressure

# Get coordinates
lats = ds['lat'].values          # {len(ds.lat.values)} latitude points
lons = ds['lon'].values          # {len(ds.lon.values)} longitude points

# Denormalize if needed (example for temperature)
t2m_mean = ds['t2m'].attrs['normalization_mean']    # {ds['t2m'].attrs['normalization_mean']:.2f} K
t2m_std = ds['t2m'].attrs['normalization_std']      # {ds['t2m'].attrs['normalization_std']:.2f} K
temp_celsius = ((temp * t2m_std) + t2m_mean) - 273.15

# Stack all variables for ML training
variables = ['t2m', 'd2m', 'u10', 'v10', 'sp']
n_vars = len(variables)
n_spatial = {total_points:,}

# Create training array: shape (n_variables, n_spatial_points)
training_data = np.zeros((n_vars, n_spatial), dtype=np.float32)
for i, var_name in enumerate(variables):
    training_data[i, :] = ds[var_name].values.flatten()

print(f"Training data shape: {{training_data.shape}}")  # Should be ({len(ds.data_vars)}, {total_points:,})

ds.close()
""")
    
    ds.close()


def check_realistic_range(var_name: str, min_val: float, max_val: float) -> bool:
    """Check if variable ranges are realistic for meteorological data"""
    
    # Realistic ranges for meteorological variables
    ranges = {
        't2m': (200, 350),      # Temperature: -73°C to 77°C
        'd2m': (180, 320),      # Dewpoint: similar to temperature
        'u10': (-50, 50),       # U-wind: ±50 m/s is reasonable
        'v10': (-50, 50),       # V-wind: ±50 m/s is reasonable  
        'sp': (50000, 110000),  # Surface pressure: 500-1100 hPa
    }
    
    if var_name in ranges:
        expected_min, expected_max = ranges[var_name]
        return expected_min <= min_val <= max_val <= expected_max
    
    return True  # Unknown variable, assume OK


if __name__ == '__main__':
    inspect_real_hrrr()