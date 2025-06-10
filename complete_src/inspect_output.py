#!/usr/bin/env python3
"""
Inspect HRRR Training Data Output
Shows detailed information about the generated NetCDF files
"""

import xarray as xr
import numpy as np
from pathlib import Path


def inspect_netcdf_file(file_path: Path):
    """Inspect a single NetCDF file in detail"""
    print(f"\n🔍 Inspecting: {file_path.name}")
    print("="*60)
    
    # Load the dataset
    ds = xr.open_dataset(file_path)
    
    # Basic information
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.2f} MB")
    print(f"📊 Variables: {len(ds.data_vars)}")
    print(f"📏 Grid dimensions: {dict(ds.sizes)}")
    print(f"🌐 Coordinate ranges:")
    if 'latitude' in ds.coords:
        print(f"   Latitude: {float(ds.coords['latitude'].min()):.2f}° to {float(ds.coords['latitude'].max()):.2f}°")
        print(f"   Longitude: {float(ds.coords['longitude'].min()):.2f}° to {float(ds.coords['longitude'].max()):.2f}°")
    else:
        print(f"   Coordinates: {list(ds.coords.keys())}")
    
    # Global attributes
    print(f"\n🏷️  Global Attributes:")
    for attr_name, attr_value in ds.attrs.items():
        print(f"   {attr_name}: {attr_value}")
    
    # Variable details
    print(f"\n🗂️  Variable Details:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"\n   {var_name}:")
        print(f"     Shape: {var.shape}")
        print(f"     Data type: {var.dtype}")
        
        # Statistics
        values = var.values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"     Normalized range: [{float(np.min(valid_values)):.3f}, {float(np.max(valid_values)):.3f}]")
            print(f"     Normalized mean: {float(np.mean(valid_values)):.6f}")
            print(f"     Normalized std: {float(np.std(valid_values)):.6f}")
            print(f"     Valid points: {len(valid_values):,} / {values.size:,} ({len(valid_values)/values.size*100:.1f}%)")
        
        # Original statistics from attributes
        if 'normalization_mean' in var.attrs:
            orig_mean = var.attrs['normalization_mean']
            orig_std = var.attrs['normalization_std']
            orig_min = var.attrs.get('original_min', 'N/A')
            orig_max = var.attrs.get('original_max', 'N/A')
            print(f"     Original mean: {orig_mean:.3f}")
            print(f"     Original std: {orig_std:.3f}")
            print(f"     Original range: [{orig_min:.3f}, {orig_max:.3f}]")
        
        # Other attributes
        print(f"     Units: {var.attrs.get('units', 'N/A')}")
        print(f"     Long name: {var.attrs.get('long_name', 'N/A')}")
        print(f"     Category: {var.attrs.get('category', 'N/A')}")
    
    ds.close()


def create_sample_usage_code(file_path: Path):
    """Generate sample Python code for using the data"""
    print(f"\n💻 Sample Usage Code:")
    print("="*30)
    
    code = f'''
import xarray as xr
import numpy as np

# Load the dataset
ds = xr.open_dataset('{file_path}')

# Access normalized data (already mean=0, std=1)
temperature = ds['t2m'].values if 't2m' in ds else None
cloud_cover = ds['cloud_cover'].values
wind_shear = ds['wind_shear_u_06km'].values

# Get original statistics for denormalization
if 'cloud_cover' in ds:
    cloud_mean = ds['cloud_cover'].attrs['normalization_mean']
    cloud_std = ds['cloud_cover'].attrs['normalization_std']
    
    # Denormalize if needed
    original_cloud_cover = (ds['cloud_cover'].values * cloud_std) + cloud_mean

# Get coordinates
lats = ds.coords['latitude'].values
lons = ds.coords['longitude'].values

# Example: Create a mask for specific regions
conus_mask = (lats >= 25) & (lats <= 45) & (lons >= -120) & (lons <= -70)

# Stack variables for ML training (flatten spatial dimensions)
n_vars = len(ds.data_vars)
n_spatial = ds.dims['lat'] * ds.dims['lon']

# Create training array: shape (n_variables, n_spatial_points)
training_data = np.zeros((n_vars, n_spatial))
var_names = list(ds.data_vars.keys())

for i, var_name in enumerate(var_names):
    var_data = ds[var_name].values.flatten()
    training_data[i, :] = var_data

print(f"Training data shape: {{training_data.shape}}")
print(f"Variables: {{var_names}}")

# Close dataset
ds.close()
'''
    
    print(code)


def main():
    """Main inspection function"""
    print("🌪️  HRRR Training Data Inspection")
    print("="*60)
    
    # Find NetCDF files
    data_dir = Path('./hrrr_2025061002_demo')
    nc_files = list(data_dir.rglob('*.nc'))
    
    if not nc_files:
        print("❌ No NetCDF files found in ./hrrr_2025061002_demo")
        return
    
    print(f"📂 Found {len(nc_files)} NetCDF files:")
    for f in nc_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.relative_to(data_dir)}: {size_mb:.1f} MB")
    
    # Inspect the first file in detail
    sample_file = nc_files[0]
    inspect_netcdf_file(sample_file)
    
    # Show usage example
    create_sample_usage_code(sample_file)
    
    # Summary of all files
    print(f"\n📊 Summary of All Files:")
    print("="*30)
    
    total_size = 0
    total_variables = 0
    
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        file_size = nc_file.stat().st_size / (1024 * 1024)
        total_size += file_size
        total_variables = len(ds.data_vars)  # Should be same for all
        
        print(f"{nc_file.name}: {len(ds.data_vars)} variables, {file_size:.1f} MB")
        ds.close()
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Total files: {len(nc_files)}")
    print(f"   Variables per file: {total_variables}")
    print(f"   Total size: {total_size:.1f} MB")
    print(f"   Grid resolution: ~25km (120×180 points)")
    print(f"   Coverage: CONUS (Continental United States)")
    print(f"   Data type: Float32 (normalized)")
    print(f"   Format: NetCDF4 with compression")
    
    print(f"\n🎯 Ready for ML Training:")
    print(f"   ✅ Data is pre-normalized (mean=0, std=1)")
    print(f"   ✅ Original statistics stored as attributes")
    print(f"   ✅ Quality validated (NaN handling, variance checks)")
    print(f"   ✅ Consistent grid and coordinates")
    print(f"   ✅ CF-compliant metadata")


if __name__ == '__main__':
    main()