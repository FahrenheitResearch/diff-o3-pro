#!/usr/bin/env python3
"""
HRRR Training Data Pipeline
Generates structured numerical datasets for diffusion model training

This script adapts the HRRR visualization system into a robust numerical data 
processing pipeline suitable for training diffusion-based machine learning models.
It produces structured numerical datasets in NetCDF format instead of PNG visualizations.
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import xarray as xr

# Import the existing HRRR processing infrastructure
sys.path.append(str(Path(__file__).parent.parent))
from hrrr_processor_refactored import HRRRProcessor
from field_registry import FieldRegistry
from derived_params import compute_derived_parameter

warnings.filterwarnings('ignore')


class HRRRTrainingPipeline:
    """
    HRRR Training Data Pipeline for ML model preparation
    
    Converts HRRR GRIB data into normalized NetCDF datasets suitable for 
    diffusion model training, replacing visualization with numerical serialization.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the training pipeline
        
        Args:
            config_dir: Directory containing parameter configuration files
        """
        self.processor = HRRRProcessor(config_dir)
        self.registry = FieldRegistry(config_dir)
        
        # Load all field configurations once
        print("🔄 Loading field configurations for training pipeline...")
        self.all_fields = self.registry.load_all_fields()
        
        if not self.all_fields:
            raise RuntimeError("❌ Failed to load field configurations")
        
        print(f"✅ Loaded {len(self.all_fields)} field configurations")
        
    def select_training_variables(self, 
                                include_categories: Optional[List[str]] = None,
                                exclude_categories: Optional[List[str]] = None,
                                include_derived: bool = True,
                                max_variables: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Select variables suitable for ML training
        
        Args:
            include_categories: Categories to include (None = all)
            exclude_categories: Categories to exclude
            include_derived: Whether to include derived parameters
            max_variables: Maximum number of variables to select
            
        Returns:
            Dictionary of selected field configurations
        """
        selected_fields = {}
        
        for field_name, field_config in self.all_fields.items():
            # Skip if derived parameters not wanted
            if not include_derived and field_config.get('derived', False):
                continue
                
            # Category filtering
            category = field_config.get('category', 'unknown')
            
            if include_categories and category not in include_categories:
                continue
                
            if exclude_categories and category in exclude_categories:
                continue
                
            selected_fields[field_name] = field_config
            
            # Limit number of variables if specified
            if max_variables and len(selected_fields) >= max_variables:
                break
        
        print(f"📊 Selected {len(selected_fields)} variables for training:")
        categories = {}
        for config in selected_fields.values():
            cat = config.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} variables")
            
        return selected_fields
    
    def load_field_data_for_training(self, 
                                   field_name: str, 
                                   field_config: Dict[str, Any],
                                   grib_file: Path,
                                   wrfsfc_file: Optional[Path] = None) -> Optional[xr.DataArray]:
        """Load field data without any visualization processing
        
        Args:
            field_name: Name of the field
            field_config: Field configuration
            grib_file: Path to GRIB file
            wrfsfc_file: Optional path to wrfsfc file for specific fields
            
        Returns:
            xarray DataArray with raw numerical data
        """
        try:
            print(f"  Loading: {field_name}")
            
            # Check if this is a derived parameter
            if field_config.get('derived'):
                # Load derived parameter
                data = self.processor.load_derived_parameter(
                    field_name, field_config, grib_file, wrfsfc_file
                )
            else:
                # Choose appropriate file for this field
                if field_config.get('category') == 'smoke' and wrfsfc_file:
                    # Try wrfsfc first for smoke fields
                    data = self.processor.load_field_data(wrfsfc_file, field_name, field_config)
                    if data is None:
                        data = self.processor.load_field_data(grib_file, field_name, field_config)
                elif field_config.get('category') == 'updraft_helicity' and wrfsfc_file:
                    # Handle updraft helicity fields
                    if 'layer' in field_config:
                        top, bottom = map(int, field_config['layer'].split('-'))
                        data = self.processor.load_uh_layer(wrfsfc_file, top, bottom)
                    else:
                        data = self.processor.load_field_data(wrfsfc_file, field_name, field_config)
                    
                    if data is None:
                        data = self.processor.load_field_data(grib_file, field_name, field_config)
                else:
                    # Use standard GRIB file
                    data = self.processor.load_field_data(grib_file, field_name, field_config)
            
            if data is None:
                print(f"    ❌ Failed to load {field_name}")
                return None
                
            # Ensure data is 2D and has proper coordinates
            if len(data.dims) > 2:
                # Remove extra dimensions
                for dim in data.dims:
                    if dim not in ['latitude', 'longitude', 'y', 'x'] and data.sizes[dim] == 1:
                        data = data.squeeze(dim)
                    elif dim not in ['latitude', 'longitude', 'y', 'x']:
                        data = data.isel({dim: 0})
                        
            print(f"    ✅ Loaded {field_name}: shape {data.shape}")
            return data
            
        except Exception as e:
            print(f"    ❌ Error loading {field_name}: {e}")
            return None
    
    def validate_data_quality(self, data: xr.DataArray, field_name: str) -> bool:
        """Validate data quality before including in dataset
        
        Args:
            data: Data array to validate
            field_name: Name of the field for logging
            
        Returns:
            True if data passes quality checks
        """
        try:
            # Check for all NaN
            if np.all(np.isnan(data.values)):
                print(f"    ⚠️ {field_name}: All NaN values")
                return False
                
            # Check for all zeros (might indicate missing data)
            if np.all(data.values == 0):
                print(f"    ⚠️ {field_name}: All zero values")
                return False
                
            # Check for reasonable data range (no extreme outliers)
            valid_data = data.values[~np.isnan(data.values)]
            if len(valid_data) == 0:
                print(f"    ⚠️ {field_name}: No valid data")
                return False
                
            # Check standard deviation (data should have some variability)
            if np.std(valid_data) < 1e-10:
                print(f"    ⚠️ {field_name}: No variability (std < 1e-10)")
                return False
                
            print(f"    ✅ {field_name}: Quality check passed")
            return True
            
        except Exception as e:
            print(f"    ❌ {field_name}: Validation error: {e}")
            return False
    
    def normalize_data(self, data: xr.DataArray, field_name: str) -> xr.DataArray:
        """Normalize data with explicit mean/std tracking
        
        Args:
            data: Input data array
            field_name: Name of the field
            
        Returns:
            Normalized data array with mean/std stored in attributes
        """
        try:
            # Calculate statistics on valid data only
            valid_mask = ~np.isnan(data.values)
            valid_data = data.values[valid_mask]
            
            if len(valid_data) == 0:
                raise ValueError("No valid data for normalization")
                
            mean_val = float(np.mean(valid_data))
            std_val = float(np.std(valid_data))
            
            # Handle near-zero standard deviation
            if std_val < 1e-6:
                print(f"    ⚠️ {field_name}: Low std ({std_val}), using std=1.0")
                std_val = 1.0
            
            # Normalize the data
            normalized_data = (data - mean_val) / std_val
            
            # Store normalization parameters as attributes
            normalized_data.attrs.update({
                'normalization_mean': mean_val,
                'normalization_std': std_val,
                'original_min': float(np.min(valid_data)),
                'original_max': float(np.max(valid_data)),
                'valid_points': int(np.sum(valid_mask)),
                'total_points': int(data.size),
                'field_name': field_name
            })
            
            print(f"    📊 {field_name}: normalized (mean={mean_val:.3f}, std={std_val:.3f})")
            return normalized_data
            
        except Exception as e:
            print(f"    ❌ {field_name}: Normalization error: {e}")
            raise
    
    def create_training_dataset(self,
                              fields_data: Dict[str, xr.DataArray],
                              cycle: str,
                              forecast_hour: int,
                              metadata: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """Create structured xarray Dataset for training
        
        Args:
            fields_data: Dictionary of loaded and validated field data
            cycle: HRRR cycle (YYYYMMDDHH)
            forecast_hour: Forecast hour
            metadata: Additional metadata to include
            
        Returns:
            Structured xarray Dataset ready for NetCDF serialization
        """
        print(f"  🔧 Assembling dataset with {len(fields_data)} variables...")
        
        # Get reference coordinates from first field
        reference_field = next(iter(fields_data.values()))
        coords = reference_field.coords
        
        # Verify all fields have consistent coordinates
        for field_name, field_data in fields_data.items():
            if not np.array_equal(field_data.latitude.values, reference_field.latitude.values):
                raise ValueError(f"Latitude coordinates mismatch for field: {field_name}")
            if not np.array_equal(field_data.longitude.values, reference_field.longitude.values):
                raise ValueError(f"Longitude coordinates mismatch for field: {field_name}")
        
        # Create data variables dictionary
        data_vars = {}
        for field_name, field_data in fields_data.items():
            # Normalize the data
            normalized_data = self.normalize_data(field_data, field_name)
            
            # Store with consistent dimensions
            data_vars[field_name] = (('lat', 'lon'), normalized_data.values)
        
        # Create the dataset
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                'lat': ('lat', reference_field.latitude.values),
                'lon': ('lon', reference_field.longitude.values)
            }
        )
        
        # Add global attributes
        cycle_dt = datetime.strptime(cycle, '%Y%m%d%H')
        valid_dt = cycle_dt + timedelta(hours=forecast_hour)
        
        dataset.attrs.update({
            'title': 'HRRR Training Dataset',
            'description': 'Normalized numerical data for diffusion model training',
            'source': 'High-Resolution Rapid Refresh (HRRR) Model',
            'cycle': cycle,
            'forecast_hour': forecast_hour,
            'init_time': cycle_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'valid_time': valid_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'created': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'creator': 'HRRR Training Pipeline',
            'num_variables': len(fields_data),
            'grid_shape': f"{reference_field.shape[0]}x{reference_field.shape[1]}",
            'conventions': 'CF-1.8'
        })
        
        # Add field-specific metadata as variable attributes
        for field_name, field_data in fields_data.items():
            if field_name in dataset.data_vars:
                # Copy normalization attributes
                dataset[field_name].attrs.update(field_data.attrs)
                
                # Add field configuration metadata (NetCDF compatible)
                field_config = self.all_fields.get(field_name, {})
                dataset[field_name].attrs.update({
                    'long_name': field_config.get('title', field_name),
                    'units': field_config.get('units', 'dimensionless'),
                    'category': field_config.get('category', 'unknown'),
                    'derived': str(field_config.get('derived', False))  # Convert bool to string
                })
        
        if metadata:
            # Convert metadata to NetCDF-compatible types
            compatible_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, bool):
                    compatible_metadata[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    compatible_metadata[key] = str(value)
                elif isinstance(value, dict):
                    compatible_metadata[key] = str(value)
                else:
                    compatible_metadata[key] = value
            dataset.attrs.update(compatible_metadata)
        
        print(f"  ✅ Dataset created: {len(fields_data)} variables, shape {dataset.dims}")
        return dataset
    
    def save_training_dataset(self,
                            dataset: xr.Dataset,
                            output_path: Path,
                            compression_level: int = 4) -> None:
        """Save dataset to NetCDF with proper compression
        
        Args:
            dataset: xarray Dataset to save
            output_path: Output file path
            compression_level: Compression level (0-9)
        """
        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure compression for all variables
            encoding = {}
            for var_name in dataset.data_vars:
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'dtype': 'float32'  # Use float32 to save space
                }
            
            # Save to NetCDF
            print(f"  💾 Saving dataset to: {output_path}")
            dataset.to_netcdf(
                output_path,
                encoding=encoding,
                format='NETCDF4'
            )
            
            # Verify file was created and get size
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"    ✅ Saved successfully ({file_size_mb:.1f} MB)")
            else:
                raise RuntimeError("File was not created")
                
        except Exception as e:
            print(f"    ❌ Save failed: {e}")
            raise
        finally:
            # Clean up dataset to free memory
            dataset.close()
    
    def generate_training_data(self,
                             cycles: List[str],
                             forecast_hours: List[int],
                             output_base_dir: Path,
                             variables: Optional[Dict[str, Dict[str, Any]]] = None,
                             max_variables: Optional[int] = None) -> Dict[str, Any]:
        """Generate training data for multiple cycles and forecast hours
        
        Args:
            cycles: List of HRRR cycles (YYYYMMDDHH format)
            forecast_hours: List of forecast hours to process
            output_base_dir: Base output directory
            variables: Variables to process (None = auto-select)
            max_variables: Maximum number of variables to process
            
        Returns:
            Summary statistics of the generation process
        """
        print(f"🚀 Starting training data generation")
        print(f"  Cycles: {len(cycles)} ({cycles[0]} to {cycles[-1]})")
        print(f"  Forecast hours: {forecast_hours}")
        print(f"  Output directory: {output_base_dir}")
        
        # Select variables if not provided
        if variables is None:
            variables = self.select_training_variables(
                include_categories=['severe', 'instability', 'atmospheric', 'surface'],
                exclude_categories=['smoke'],  # Exclude smoke for now (memory intensive)
                include_derived=True,
                max_variables=max_variables
            )
        
        print(f"  Variables: {len(variables)}")
        
        # Initialize tracking
        total_datasets = 0
        successful_datasets = 0
        failed_datasets = 0
        processing_times = []
        
        # Process each cycle and forecast hour
        for cycle in cycles:
            for fh in forecast_hours:
                cycle_start_time = time.time()
                print(f"\n📅 Processing cycle {cycle}, F{fh:02d}")
                
                try:
                    # Download/load GRRR files
                    grib_file = self.processor.download_hrrr_file(
                        cycle, fh, output_base_dir / 'grib_cache', 'wrfprs'
                    )
                    
                    wrfsfc_file = None
                    if any(cfg.get('category') in ['smoke', 'updraft_helicity'] for cfg in variables.values()):
                        wrfsfc_file = self.processor.download_hrrr_file(
                            cycle, fh, output_base_dir / 'grib_cache', 'wrfsfc'
                        )
                    
                    if not grib_file or not grib_file.exists():
                        print(f"  ❌ Failed to download GRIB files for {cycle} F{fh:02d}")
                        failed_datasets += 1
                        continue
                    
                    # Load all field data
                    fields_data = {}
                    for field_name, field_config in variables.items():
                        data = self.load_field_data_for_training(
                            field_name, field_config, grib_file, wrfsfc_file
                        )
                        
                        if data is not None and self.validate_data_quality(data, field_name):
                            fields_data[field_name] = data
                        else:
                            print(f"    ⚠️ Skipping {field_name} due to quality issues")
                    
                    if not fields_data:
                        print(f"  ❌ No valid fields loaded for {cycle} F{fh:02d}")
                        failed_datasets += 1
                        continue
                    
                    # Create dataset
                    dataset = self.create_training_dataset(
                        fields_data, cycle, fh,
                        metadata={'processing_time': time.time() - cycle_start_time}
                    )
                    
                    # Save dataset
                    output_path = output_base_dir / f"cycle_{cycle}" / f"forecast_hour_F{fh:02d}.nc"
                    self.save_training_dataset(dataset, output_path)
                    
                    # Track success
                    processing_time = time.time() - cycle_start_time
                    processing_times.append(processing_time)
                    successful_datasets += 1
                    
                    print(f"  ✅ Completed in {processing_time:.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {cycle} F{fh:02d}: {e}")
                    failed_datasets += 1
                
                total_datasets += 1
        
        # Generate summary
        summary = {
            'total_datasets': total_datasets,
            'successful_datasets': successful_datasets,
            'failed_datasets': failed_datasets,
            'success_rate': successful_datasets / total_datasets if total_datasets > 0 else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_processing_time': sum(processing_times),
            'output_directory': str(output_base_dir),
            'variables_processed': list(variables.keys()),
            'num_variables': len(variables)
        }
        
        print(f"\n" + "="*60)
        print(f"📊 TRAINING DATA GENERATION SUMMARY")
        print(f"="*60)
        print(f"✅ Successful: {successful_datasets}/{total_datasets} ({summary['success_rate']*100:.1f}%)")
        print(f"❌ Failed: {failed_datasets}")
        print(f"⏱️  Average time per dataset: {summary['avg_processing_time']:.1f}s")
        print(f"🎯 Variables per dataset: {summary['num_variables']}")
        print(f"📁 Output directory: {output_base_dir}")
        
        return summary


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate HRRR training datasets for diffusion models'
    )
    parser.add_argument('--cycles', nargs='+', required=True,
                       help='HRRR cycles to process (YYYYMMDDHH format)')
    parser.add_argument('--forecast-hours', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='Forecast hours to process')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for training datasets')
    parser.add_argument('--max-variables', type=int,
                       help='Maximum number of variables to include')
    parser.add_argument('--categories', nargs='+', 
                       default=['severe', 'instability', 'atmospheric', 'surface'],
                       help='Categories to include')
    parser.add_argument('--exclude-derived', action='store_true',
                       help='Exclude derived parameters')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = HRRRTrainingPipeline()
    
    # Select variables
    variables = pipeline.select_training_variables(
        include_categories=args.categories,
        include_derived=not args.exclude_derived,
        max_variables=args.max_variables
    )
    
    # Generate training data
    summary = pipeline.generate_training_data(
        cycles=args.cycles,
        forecast_hours=args.forecast_hours,
        output_base_dir=args.output_dir,
        variables=variables
    )
    
    # Save summary
    summary_file = args.output_dir / 'generation_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()