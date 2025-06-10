#!/usr/bin/env python3
"""
Test script for the HRRR Training Pipeline
Validates functionality with a minimal dataset
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def test_pipeline_basic_functionality():
    """Test basic pipeline functionality with minimal configuration"""
    print("🧪 Testing HRRR Training Pipeline")
    print("="*50)
    
    try:
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = HRRRTrainingPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Test variable selection
        print("\n2. Testing variable selection...")
        variables = pipeline.select_training_variables(
            include_categories=['surface', 'atmospheric'],
            include_derived=False,  # Start with non-derived for simplicity
            max_variables=5  # Limit to just 5 variables for testing
        )
        print(f"✅ Selected {len(variables)} variables: {list(variables.keys())}")
        
        # Test with a cycle that should have data available
        print("\n3. Testing data generation...")
        now = datetime.utcnow()
        # Use a cycle from 12-24 hours ago to ensure data availability
        test_cycle = (now - timedelta(hours=18)).strftime('%Y%m%d%H')
        
        print(f"📅 Using test cycle: {test_cycle}")
        print(f"📊 Testing with forecast hour: F00")
        
        # Create output directory for test
        output_dir = Path('./test_output')
        output_dir.mkdir(exist_ok=True)
        
        # Generate a single dataset for testing
        summary = pipeline.generate_training_data(
            cycles=[test_cycle],
            forecast_hours=[0],  # Just F00 for quick test
            output_base_dir=output_dir,
            variables=variables
        )
        
        print(f"✅ Test completed successfully!")
        print(f"📊 Summary: {summary['successful_datasets']}/{summary['total_datasets']} datasets generated")
        
        # Verify output file exists
        expected_file = output_dir / f"cycle_{test_cycle}" / "forecast_hour_F00.nc"
        if expected_file.exists():
            file_size_mb = expected_file.stat().st_size / (1024 * 1024)
            print(f"📁 Output file created: {expected_file} ({file_size_mb:.1f} MB)")
            
            # Try to read the file back to verify it's valid NetCDF
            import xarray as xr
            try:
                test_ds = xr.open_dataset(expected_file)
                print(f"📊 Dataset verification:")
                print(f"   Variables: {len(test_ds.data_vars)}")
                print(f"   Dimensions: {dict(test_ds.dims)}")
                print(f"   Attributes: {len(test_ds.attrs)} global attributes")
                
                # Check normalization attributes
                for var_name in list(test_ds.data_vars.keys())[:3]:  # Check first 3 variables
                    if 'normalization_mean' in test_ds[var_name].attrs:
                        mean = test_ds[var_name].attrs['normalization_mean']
                        std = test_ds[var_name].attrs['normalization_std']
                        print(f"   {var_name}: normalized (mean={mean:.3f}, std={std:.3f})")
                
                test_ds.close()
                print("✅ NetCDF file is valid and properly structured")
                
                return True
                
            except Exception as e:
                print(f"❌ Error reading NetCDF file: {e}")
                return False
        else:
            print(f"❌ Expected output file not found: {expected_file}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_selection():
    """Test variable selection functionality"""
    print("\n🧪 Testing Variable Selection")
    print("="*30)
    
    try:
        pipeline = HRRRTrainingPipeline()
        
        # Test different selection scenarios
        print("1. Testing category filtering...")
        severe_vars = pipeline.select_training_variables(
            include_categories=['severe'],
            max_variables=10
        )
        print(f"   Severe weather variables: {len(severe_vars)}")
        
        print("2. Testing derived parameter inclusion...")
        derived_vars = pipeline.select_training_variables(
            include_categories=['severe'],
            include_derived=True,
            max_variables=5
        )
        has_derived = any(cfg.get('derived', False) for cfg in derived_vars.values())
        print(f"   Has derived parameters: {has_derived}")
        
        print("3. Testing variable limits...")
        limited_vars = pipeline.select_training_variables(max_variables=3)
        print(f"   Limited to 3 variables: {len(limited_vars)} selected")
        
        print("✅ Variable selection tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Variable selection test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 HRRR Training Pipeline Test Suite")
    print("="*60)
    
    # Run tests
    tests = [
        ("Variable Selection", test_variable_selection),
        ("Basic Pipeline Functionality", test_pipeline_basic_functionality)
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
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())