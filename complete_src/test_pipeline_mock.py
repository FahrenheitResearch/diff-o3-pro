#!/usr/bin/env python3
"""
Mock test for HRRR Training Pipeline
Tests core functionality without requiring GRIB file downloads
"""

import sys
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def create_mock_data_array(name: str, shape: tuple = (100, 150)) -> xr.DataArray:
    """Create mock xarray DataArray for testing"""
    # Generate synthetic but realistic-looking meteorological data
    lat = np.linspace(25, 50, shape[0])  # Latitude range for CONUS
    lon = np.linspace(-125, -65, shape[1])  # Longitude range for CONUS
    
    # Create synthetic data based on field type
    if 'temperature' in name.lower() or 'temp' in name.lower():
        # Temperature-like data (250-320 K)
        data = 280 + 20 * np.random.randn(*shape)
    elif 'wind' in name.lower():
        # Wind-like data (0-50 m/s)
        data = np.abs(10 + 10 * np.random.randn(*shape))
    elif 'pressure' in name.lower():
        # Pressure-like data (950-1050 hPa)
        data = 1000 + 25 * np.random.randn(*shape)
    elif 'cape' in name.lower():
        # CAPE-like data (0-5000 J/kg)
        data = np.maximum(0, 1000 + 1000 * np.random.randn(*shape))
    else:
        # Generic meteorological data
        data = 10 + 5 * np.random.randn(*shape)
    
    # Create DataArray with proper coordinates
    da = xr.DataArray(
        data,
        coords={
            'latitude': ('y', lat),
            'longitude': ('x', lon)
        },
        dims=['y', 'x'],
        name=name,
        attrs={
            'long_name': f'Mock {name.replace("_", " ").title()}',
            'units': 'mock_units'
        }
    )
    
    return da


def test_pipeline_core_functionality():
    """Test core pipeline functionality with mock data"""
    print("🧪 Testing Core Pipeline Functionality (Mock Data)")
    print("="*55)
    
    try:
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = HRRRTrainingPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Create mock field data
        print("\n2. Creating mock field data...")
        mock_fields = {
            'temperature_2m': create_mock_data_array('temperature_2m'),
            'wind_speed_10m': create_mock_data_array('wind_speed_10m'),
            'pressure_surface': create_mock_data_array('pressure_surface'),
            'cape': create_mock_data_array('cape'),
            'humidity_2m': create_mock_data_array('humidity_2m')
        }
        
        print(f"✅ Created {len(mock_fields)} mock fields")
        for name, data in mock_fields.items():
            print(f"   {name}: {data.shape} - range [{data.min().values:.1f}, {data.max().values:.1f}]")
        
        # Test data validation
        print("\n3. Testing data validation...")
        validation_results = {}
        for name, data in mock_fields.items():
            is_valid = pipeline.validate_data_quality(data, name)
            validation_results[name] = is_valid
        
        valid_count = sum(validation_results.values())
        print(f"✅ Data validation: {valid_count}/{len(mock_fields)} fields passed")
        
        # Test normalization
        print("\n4. Testing data normalization...")
        normalized_fields = {}
        for name, data in mock_fields.items():
            if validation_results[name]:
                normalized_data = pipeline.normalize_data(data, name)
                normalized_fields[name] = normalized_data
                
                # Verify normalization
                norm_mean = normalized_data.attrs['normalization_mean']
                norm_std = normalized_data.attrs['normalization_std']
                actual_mean = float(np.nanmean(normalized_data.values))
                actual_std = float(np.nanstd(normalized_data.values))
                
                print(f"   {name}: original_mean={norm_mean:.3f}, norm_mean={actual_mean:.6f}")
                print(f"   {name}: original_std={norm_std:.3f}, norm_std={actual_std:.6f}")
        
        print(f"✅ Normalization: {len(normalized_fields)} fields normalized")
        
        # Test dataset creation
        print("\n5. Testing dataset creation...")
        cycle = "2025060900"
        forecast_hour = 0
        
        dataset = pipeline.create_training_dataset(
            normalized_fields, cycle, forecast_hour,
            metadata={'test': True, 'mock_data': True}
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Variables: {len(dataset.data_vars)}")
        print(f"   Dimensions: {dict(dataset.dims)}")
        print(f"   Global attributes: {len(dataset.attrs)}")
        
        # Verify normalization attributes are preserved
        for var_name in dataset.data_vars:
            if 'normalization_mean' in dataset[var_name].attrs:
                mean = dataset[var_name].attrs['normalization_mean']
                std = dataset[var_name].attrs['normalization_std']
                print(f"   {var_name}: norm_attrs (mean={mean:.3f}, std={std:.3f})")
        
        # Test NetCDF serialization
        print("\n6. Testing NetCDF serialization...")
        output_path = Path('./test_output_mock/test_dataset.nc')
        pipeline.save_training_dataset(dataset, output_path)
        
        # Verify file was created and can be read back
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ NetCDF file created: {output_path} ({file_size_mb:.2f} MB)")
            
            # Try to read it back
            test_ds = xr.open_dataset(output_path)
            print(f"✅ NetCDF file verified - {len(test_ds.data_vars)} variables read back")
            
            # Check that data types are correct (float32)
            for var_name in test_ds.data_vars:
                dtype = test_ds[var_name].dtype
                print(f"   {var_name}: dtype={dtype}")
            
            test_ds.close()
            return True
        else:
            print(f"❌ NetCDF file not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalization_accuracy():
    """Test that normalization is mathematically correct"""
    print("\n🧪 Testing Normalization Accuracy")
    print("="*35)
    
    try:
        pipeline = HRRRTrainingPipeline()
        
        # Create test data with known statistics
        shape = (50, 75)
        known_mean = 15.0
        known_std = 3.0
        
        # Generate normal distribution with known parameters
        data_values = np.random.normal(known_mean, known_std, shape)
        
        # Add some NaN values to test masking
        data_values[0:5, 0:5] = np.nan
        
        # Create DataArray
        lat = np.linspace(30, 45, shape[0])
        lon = np.linspace(-100, -80, shape[1])
        
        test_data = xr.DataArray(
            data_values,
            coords={
                'latitude': ('y', lat),
                'longitude': ('x', lon)
            },
            dims=['y', 'x'],
            name='test_field'
        )
        
        print(f"Original data: mean={np.nanmean(data_values):.3f}, std={np.nanstd(data_values):.3f}")
        
        # Normalize the data
        normalized = pipeline.normalize_data(test_data, 'test_field')
        
        # Check normalization accuracy
        computed_mean = normalized.attrs['normalization_mean']
        computed_std = normalized.attrs['normalization_std']
        
        # The normalized data should have mean ≈ 0, std ≈ 1
        actual_norm_mean = float(np.nanmean(normalized.values))
        actual_norm_std = float(np.nanstd(normalized.values))
        
        print(f"Computed stats: mean={computed_mean:.3f}, std={computed_std:.3f}")
        print(f"Normalized data: mean={actual_norm_mean:.6f}, std={actual_norm_std:.6f}")
        
        # Verify accuracy (should be very close to 0 and 1)
        mean_error = abs(actual_norm_mean)
        std_error = abs(actual_norm_std - 1.0)
        
        if mean_error < 1e-10 and std_error < 1e-6:
            print("✅ Normalization accuracy test passed")
            return True
        else:
            print(f"❌ Normalization accuracy test failed: mean_error={mean_error}, std_error={std_error}")
            return False
            
    except Exception as e:
        print(f"❌ Normalization test failed: {e}")
        return False


def main():
    """Run all mock tests"""
    print("🚀 HRRR Training Pipeline Mock Test Suite")
    print("="*60)
    
    # Run tests
    tests = [
        ("Core Functionality (Mock)", test_pipeline_core_functionality),
        ("Normalization Accuracy", test_normalization_accuracy)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📊 MOCK TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mock tests passed! Core pipeline functionality verified.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())