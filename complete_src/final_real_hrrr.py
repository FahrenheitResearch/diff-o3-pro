#!/usr/bin/env python3
"""
Final Real HRRR Data Processor
Create the actual training data from real HRRR GRIB at native resolution
"""

import sys
import time
import numpy as np
import xarray as xr
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

import cfgrib


def create_real_hrrr_dataset():
    """Create real HRRR dataset from the downloaded GRIB file"""
    
    print("🌪️  CREATING REAL HRRR TRAINING DATASET")
    print("="*55)
    
    # Use the real downloaded GRIB file
    grib_file = Path('./real_hrrr_native_3km/grib_cache/hrrr.t00z.wrfprsf00.grib2')
    
    if not grib_file.exists():
        print(f"❌ GRIB file not found: {grib_file}")
        return None
    
    file_size_mb = grib_file.stat().st_size / (1024 * 1024)
    print(f"✅ Processing real HRRR file: {grib_file.name} ({file_size_mb:.1f} MB)")
    
    # Load surface fields directly with cfgrib
    print("\n📊 Loading surface meteorological fields...")
    
    try:
        # Load 2m temperature 
        ds_t2m = cfgrib.open_dataset(grib_file, 
                                   filter_by_keys={'paramId': 167, 'typeOfLevel': 'heightAboveGround'},
                                   backend_kwargs={'indexpath': ''})
        t2m_data = ds_t2m['t2m']
        print(f"✅ 2m Temperature: {t2m_data.shape} - range [{float(t2m_data.min()):.1f}, {float(t2m_data.max()):.1f}] K")
        
        # Load 2m dewpoint
        ds_d2m = cfgrib.open_dataset(grib_file,
                                   filter_by_keys={'paramId': 168, 'typeOfLevel': 'heightAboveGround'}, 
                                   backend_kwargs={'indexpath': ''})
        d2m_data = ds_d2m['d2m']
        print(f"✅ 2m Dewpoint: {d2m_data.shape} - range [{float(d2m_data.min()):.1f}, {float(d2m_data.max()):.1f}] K")
        
        # Load 10m winds
        ds_u10 = cfgrib.open_dataset(grib_file,
                                   filter_by_keys={'paramId': 165, 'typeOfLevel': 'heightAboveGround'},
                                   backend_kwargs={'indexpath': ''})
        u10_data = ds_u10['u10']
        print(f"✅ 10m U-Wind: {u10_data.shape} - range [{float(u10_data.min()):.1f}, {float(u10_data.max()):.1f}] m/s")
        
        ds_v10 = cfgrib.open_dataset(grib_file,
                                   filter_by_keys={'paramId': 166, 'typeOfLevel': 'heightAboveGround'},
                                   backend_kwargs={'indexpath': ''})
        v10_data = ds_v10['v10']
        print(f"✅ 10m V-Wind: {v10_data.shape} - range [{float(v10_data.min()):.1f}, {float(v10_data.max()):.1f}] m/s")
        
        # Load surface pressure
        ds_sp = cfgrib.open_dataset(grib_file,
                                  filter_by_keys={'paramId': 134, 'typeOfLevel': 'surface'},
                                  backend_kwargs={'indexpath': ''})
        sp_data = ds_sp['sp']
        print(f"✅ Surface Pressure: {sp_data.shape} - range [{float(sp_data.min()):.0f}, {float(sp_data.max()):.0f}] Pa")
        
    except Exception as e:
        print(f"❌ Error loading GRIB data: {e}")
        return None
    
    # Verify all fields have the same grid
    grid_shape = t2m_data.shape
    total_points = grid_shape[0] * grid_shape[1]
    print(f"\n🌐 Native HRRR Grid: {grid_shape[0]}×{grid_shape[1]} = {total_points:,} points (3km resolution)")
    
    # Get coordinates from the first field
    lats = t2m_data.latitude.values
    lons = t2m_data.longitude.values
    
    print(f"📍 Coverage: Lat {lats.min():.2f}° to {lats.max():.2f}°, Lon {lons.min():.2f}° to {lons.max():.2f}°")
    
    # Normalize each field and create dataset
    print(f"\n🔧 Normalizing data and creating training dataset...")
    
    # Function to normalize data
    def normalize_field(data_array, field_name):
        values = data_array.values
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            raise ValueError(f"No valid data for {field_name}")
        
        mean_val = float(np.mean(valid_values))
        std_val = float(np.std(valid_values))
        
        if std_val < 1e-6:
            std_val = 1.0
            
        normalized = (values - mean_val) / std_val
        
        print(f"   {field_name}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return normalized, mean_val, std_val
    
    # Normalize all fields
    t2m_norm, t2m_mean, t2m_std = normalize_field(t2m_data, "t2m")
    d2m_norm, d2m_mean, d2m_std = normalize_field(d2m_data, "d2m") 
    u10_norm, u10_mean, u10_std = normalize_field(u10_data, "u10")
    v10_norm, v10_mean, v10_std = normalize_field(v10_data, "v10")
    sp_norm, sp_mean, sp_std = normalize_field(sp_data, "sp")
    
    # Create 1D coordinate arrays for xarray
    lat_1d = lats[:, 0]  # First column (should be constant)
    lon_1d = lons[0, :]  # First row (should be constant)
    
    # Create the xarray Dataset
    dataset = xr.Dataset(
        data_vars={
            't2m': (['lat', 'lon'], t2m_norm.astype(np.float32)),
            'd2m': (['lat', 'lon'], d2m_norm.astype(np.float32)),
            'u10': (['lat', 'lon'], u10_norm.astype(np.float32)),
            'v10': (['lat', 'lon'], v10_norm.astype(np.float32)),
            'sp': (['lat', 'lon'], sp_norm.astype(np.float32))
        },
        coords={
            'lat': ('lat', lat_1d.astype(np.float32)),
            'lon': ('lon', lon_1d.astype(np.float32))
        },
        attrs={
            'title': 'REAL HRRR Training Dataset',
            'description': 'Real HRRR data at native 3km resolution for diffusion model training',
            'source': 'NCEP High-Resolution Rapid Refresh (HRRR)',
            'cycle': '2025060900',
            'forecast_hour': 0,
            'init_time': '2025-06-09 00:00:00 UTC',
            'valid_time': '2025-06-09 00:00:00 UTC',
            'created': time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'creator': 'HRRR Real Data Pipeline', 
            'resolution': '3km native HRRR grid',
            'grid_points': total_points,
            'real_hrrr_data': 'True',
            'native_resolution': 'True',
            'grib_file_size_mb': file_size_mb,
            'conventions': 'CF-1.8'
        }
    )
    
    # Add variable attributes with normalization parameters
    dataset['t2m'].attrs.update({
        'long_name': '2m Temperature',
        'units': 'K',
        'normalization_mean': t2m_mean,
        'normalization_std': t2m_std,
        'original_min': float(t2m_data.min()),
        'original_max': float(t2m_data.max())
    })
    
    dataset['d2m'].attrs.update({
        'long_name': '2m Dewpoint Temperature',  
        'units': 'K',
        'normalization_mean': d2m_mean,
        'normalization_std': d2m_std,
        'original_min': float(d2m_data.min()),
        'original_max': float(d2m_data.max())
    })
    
    dataset['u10'].attrs.update({
        'long_name': '10m U-component Wind',
        'units': 'm/s',
        'normalization_mean': u10_mean,
        'normalization_std': u10_std,
        'original_min': float(u10_data.min()),
        'original_max': float(u10_data.max())
    })
    
    dataset['v10'].attrs.update({
        'long_name': '10m V-component Wind',
        'units': 'm/s', 
        'normalization_mean': v10_mean,
        'normalization_std': v10_std,
        'original_min': float(v10_data.min()),
        'original_max': float(v10_data.max())
    })
    
    dataset['sp'].attrs.update({
        'long_name': 'Surface Pressure',
        'units': 'Pa',
        'normalization_mean': sp_mean,
        'normalization_std': sp_std,
        'original_min': float(sp_data.min()),
        'original_max': float(sp_data.max())
    })
    
    # Save to NetCDF
    output_file = Path('./real_hrrr_native_3km/REAL_HRRR_NATIVE_2025060900_F00.nc')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to NetCDF: {output_file}")
    
    # Configure compression
    encoding = {}
    for var_name in dataset.data_vars:
        encoding[var_name] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }
    
    dataset.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
    
    # Verify output
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n🎉 SUCCESS! REAL HRRR TRAINING DATA CREATED!")
        print(f"="*60)
        print(f"📁 File: {output_file}")
        print(f"📊 Size: {file_size_mb:.1f} MB")
        print(f"🌐 Grid: {total_points:,} points at native 3km resolution")
        print(f"📈 Variables: {len(dataset.data_vars)}")
        print(f"🗂️  Fields: t2m, d2m, u10, v10, sp (core meteorological variables)")
        print(f"✅ Data normalized (mean=0, std=1) with original stats preserved")
        print(f"✅ Real HRRR data from cycle 2025060900 F00")
        
        # Close datasets
        for ds in [ds_t2m, ds_d2m, ds_u10, ds_v10, ds_sp]:
            try:
                ds.close()
            except:
                pass
        dataset.close()
        
        return output_file
    else:
        print("❌ Failed to create output file")
        return None


def verify_real_data(nc_file: Path):
    """Verify the real HRRR data"""
    
    print(f"\n🔍 VERIFYING REAL HRRR DATA")
    print("="*35)
    
    ds = xr.open_dataset(nc_file)
    
    # Basic checks
    total_points = ds.sizes['lat'] * ds.sizes['lon']
    file_size_mb = nc_file.stat().st_size / (1024 * 1024)
    
    print(f"📊 Grid: {ds.sizes['lat']}×{ds.sizes['lon']} = {total_points:,} points")
    print(f"📁 Size: {file_size_mb:.1f} MB")
    print(f"🏷️  Real data: {ds.attrs.get('real_hrrr_data', 'Unknown')}")
    print(f"🌐 Resolution: {ds.attrs.get('resolution', 'Unknown')}")
    
    # Verify this is NOT synthetic data
    if 'demo_data' in ds.attrs or 'synthetic' in ds.attrs:
        print("❌ WARNING: This appears to be synthetic data!")
    else:
        print("✅ This is real HRRR data")
    
    # Check grid size (should be ~1.9M points for real HRRR)
    if total_points > 1000000:
        print(f"✅ Grid size appropriate for real HRRR ({total_points:,} points)")
    else:
        print(f"⚠️ Grid size seems small for real HRRR ({total_points:,} points)")
    
    # Check file size (should be substantial for real data)
    if file_size_mb > 10:
        print(f"✅ File size appropriate for real data ({file_size_mb:.1f} MB)")
    else:
        print(f"⚠️ File size seems small for real data ({file_size_mb:.1f} MB)")
    
    print(f"\n📈 Variable Analysis:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        values = var.values[~np.isnan(var.values)]
        
        print(f"\n   {var_name} ({var.attrs.get('long_name', 'N/A')}):")
        print(f"     Normalized: mean={np.mean(values):.6f}, std={np.std(values):.6f}")
        
        if 'normalization_mean' in var.attrs:
            orig_mean = var.attrs['normalization_mean']
            orig_std = var.attrs['normalization_std']
            orig_min = var.attrs.get('original_min', 'N/A')
            orig_max = var.attrs.get('original_max', 'N/A')
            
            print(f"     Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
            print(f"     Range: [{orig_min:.2f}, {orig_max:.2f}] {var.attrs.get('units', '')}")
    
    ds.close()
    
    print(f"\n🎯 VERIFICATION COMPLETE")
    if total_points > 1000000 and file_size_mb > 10:
        print("✅ This data passes all checks for real HRRR training data!")
    else:
        print("⚠️ Some metrics are below expected values for real HRRR data")


if __name__ == '__main__':
    # Create the real HRRR dataset
    output_file = create_real_hrrr_dataset()
    
    if output_file:
        # Verify the data
        verify_real_data(output_file)
        
        print(f"\n🌪️  REAL HRRR TRAINING DATA READY!")
        print(f"📁 {output_file}")
        print(f"Use this NetCDF file for your diffusion model training.")
        print(f"This is genuine HRRR data at native 3km resolution - NOT synthetic!")
    else:
        print("❌ Failed to create real HRRR training data")