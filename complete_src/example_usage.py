#!/usr/bin/env python3
"""
Example Usage of HRRR Training Pipeline
Demonstrates how to generate training datasets for diffusion models
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from tools.create_training_pipeline import HRRRTrainingPipeline


def example_basic_usage():
    """Basic usage example for the training pipeline"""
    print("🚀 HRRR Training Pipeline - Basic Usage Example")
    print("="*55)
    
    # 1. Initialize the pipeline
    print("1. Initializing training pipeline...")
    pipeline = HRRRTrainingPipeline()
    
    # 2. Select variables for training
    print("\n2. Selecting variables for training...")
    
    # Example: Select severe weather parameters for ML training
    variables = pipeline.select_training_variables(
        include_categories=['severe', 'instability', 'surface'],
        include_derived=True,  # Include derived parameters like SCP, STP, etc.
        max_variables=15  # Limit to manageable number for example
    )
    
    print(f"✅ Selected {len(variables)} variables for training")
    
    # 3. Define cycles and forecast hours to process
    print("\n3. Defining processing parameters...")
    
    # Generate training data for the last 3 days, every 12 hours
    # (Adjust dates based on data availability)
    end_date = datetime.utcnow() - timedelta(days=1)  # Start from yesterday
    start_date = end_date - timedelta(days=2)  # Go back 3 days total
    
    cycles = []
    current_date = start_date.replace(hour=0)  # Start at 00Z
    while current_date <= end_date:
        cycles.append(current_date.strftime('%Y%m%d%H'))
        current_date += timedelta(hours=12)  # Every 12 hours (00Z, 12Z)
    
    forecast_hours = [0, 1, 2, 3]  # F00, F01, F02, F03
    
    print(f"📅 Processing {len(cycles)} cycles: {cycles[0]} to {cycles[-1]}")
    print(f"⏰ Forecast hours: {forecast_hours}")
    
    # 4. Set output directory
    output_dir = Path('./training_datasets')
    print(f"📁 Output directory: {output_dir}")
    
    # 5. Generate training data
    print(f"\n4. Generating training datasets...")
    print(f"   This will create {len(cycles) * len(forecast_hours)} NetCDF files")
    print(f"   Each file contains {len(variables)} normalized variables")
    
    summary = pipeline.generate_training_data(
        cycles=cycles,
        forecast_hours=forecast_hours,
        output_base_dir=output_dir,
        variables=variables
    )
    
    # 6. Display results
    print(f"\n📊 Generation Complete!")
    print(f"✅ Success rate: {summary['success_rate']*100:.1f}%")
    print(f"📁 Files created: {summary['successful_datasets']}")
    print(f"⏰ Total time: {summary['total_processing_time']:.1f} seconds")
    print(f"📂 Output location: {summary['output_directory']}")
    
    return summary


def example_custom_variable_selection():
    """Example showing custom variable selection strategies"""
    print("\n🧪 Custom Variable Selection Examples")
    print("="*40)
    
    pipeline = HRRRTrainingPipeline()
    
    # Example 1: Only surface fields (no derived parameters)
    print("1. Surface fields only (no derived):")
    surface_only = pipeline.select_training_variables(
        include_categories=['surface'],
        include_derived=False,
        max_variables=10
    )
    print(f"   Selected: {list(surface_only.keys())}")
    
    # Example 2: Severe weather focus (with derived parameters)
    print("\n2. Severe weather focus:")
    severe_focus = pipeline.select_training_variables(
        include_categories=['severe', 'instability'],
        include_derived=True,
        max_variables=20
    )
    print(f"   Selected {len(severe_focus)} variables including derived parameters")
    
    # Example 3: Exclude memory-intensive categories
    print("\n3. Efficient processing (exclude smoke/updraft helicity):")
    efficient = pipeline.select_training_variables(
        include_categories=['severe', 'instability', 'surface', 'atmospheric'],
        exclude_categories=['smoke', 'updraft_helicity'],
        include_derived=True,
        max_variables=25
    )
    print(f"   Selected {len(efficient)} variables for efficient processing")
    
    return surface_only, severe_focus, efficient


def example_data_inspection():
    """Example showing how to inspect generated datasets"""
    print("\n🔍 Dataset Inspection Example")
    print("="*35)
    
    import xarray as xr
    import numpy as np
    
    # Look for existing training datasets
    output_dir = Path('./training_datasets')
    
    if not output_dir.exists():
        print("⚠️  No training datasets found. Run basic usage example first.")
        return
    
    # Find a NetCDF file to inspect
    nc_files = list(output_dir.rglob('*.nc'))
    
    if not nc_files:
        print("⚠️  No NetCDF files found in training datasets directory.")
        return
    
    # Inspect the first file
    sample_file = nc_files[0]
    print(f"📁 Inspecting: {sample_file}")
    
    try:
        # Load the dataset
        ds = xr.open_dataset(sample_file)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Variables: {len(ds.data_vars)}")
        print(f"   Dimensions: {dict(ds.dims)}")
        print(f"   Global attributes: {len(ds.attrs)}")
        
        print(f"\n🗂️  Variables in dataset:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"   {var_name}:")
            print(f"     Shape: {var.shape}")
            print(f"     Data type: {var.dtype}")
            print(f"     Units: {var.attrs.get('units', 'N/A')}")
            
            # Check normalization
            if 'normalization_mean' in var.attrs:
                orig_mean = var.attrs['normalization_mean']
                orig_std = var.attrs['normalization_std']
                current_mean = float(np.nanmean(var.values))
                current_std = float(np.nanstd(var.values))
                print(f"     Original: mean={orig_mean:.3f}, std={orig_std:.3f}")
                print(f"     Normalized: mean={current_mean:.6f}, std={current_std:.6f}")
        
        print(f"\n🏷️  Global Attributes:")
        for attr_name, attr_value in ds.attrs.items():
            print(f"   {attr_name}: {attr_value}")
        
        # Check data quality
        print(f"\n✅ Data Quality Check:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            nan_count = np.sum(np.isnan(var.values))
            total_points = var.size
            nan_percent = (nan_count / total_points) * 100
            print(f"   {var_name}: {nan_percent:.1f}% NaN values")
        
        ds.close()
        
    except Exception as e:
        print(f"❌ Error inspecting dataset: {e}")


def example_command_line_usage():
    """Show command-line usage examples"""
    print("\n💻 Command-Line Usage Examples")
    print("="*35)
    
    print("The training pipeline can also be used from the command line:")
    print()
    
    # Generate recent 24 hours of data
    print("1. Generate training data for last 24 hours:")
    print("   python tools/create_training_pipeline.py \\")
    print("     --cycles 2025060900 2025060912 \\")
    print("     --forecast-hours 0 1 2 3 \\")
    print("     --output-dir ./training_data \\")
    print("     --categories severe instability surface")
    print()
    
    # Focus on specific weather phenomena
    print("2. Focus on severe weather (no derived parameters):")
    print("   python tools/create_training_pipeline.py \\")
    print("     --cycles 2025060900 \\")
    print("     --forecast-hours 0 1 2 \\")
    print("     --output-dir ./severe_weather_data \\")
    print("     --categories severe \\")
    print("     --exclude-derived")
    print()
    
    # Limit number of variables for faster processing
    print("3. Quick test with limited variables:")
    print("   python tools/create_training_pipeline.py \\")
    print("     --cycles 2025060900 \\")
    print("     --forecast-hours 0 \\")
    print("     --output-dir ./quick_test \\")
    print("     --max-variables 10")


def main():
    """Run all examples"""
    print("🌪️  HRRR Training Pipeline Usage Examples")
    print("="*60)
    
    try:
        # Run examples
        print("Running examples (some may take time or fail if data unavailable)...")
        
        # 1. Variable selection examples (quick)
        example_custom_variable_selection()
        
        # 2. Command line usage (informational)
        example_command_line_usage()
        
        # 3. Data inspection (if datasets exist)
        example_data_inspection()
        
        # 4. Basic usage (can take time and may fail without data)
        print(f"\n" + "="*60)
        print("Note: Basic usage example requires downloading GRIB files")
        print("This may take significant time and could fail if data is unavailable")
        print("Run manually if you want to test with real data:")
        print("  python example_usage.py --run-basic")
        
        if len(sys.argv) > 1 and '--run-basic' in sys.argv:
            example_basic_usage()
        
        print(f"\n🎉 Examples completed!")
        print(f"\nNext steps:")
        print(f"1. Customize variable selection for your use case")
        print(f"2. Run training data generation for your desired time periods")
        print(f"3. Use the generated NetCDF files with your diffusion model training")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()