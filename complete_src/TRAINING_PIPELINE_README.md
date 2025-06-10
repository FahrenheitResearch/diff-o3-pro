# HRRR Training Pipeline Documentation

This document describes the HRRR Training Data Pipeline, which converts the existing HRRR visualization system into a robust numerical data processing pipeline suitable for training diffusion-based machine learning models.

## Overview

The training pipeline transforms HRRR GRIB data into structured, normalized NetCDF datasets by:
- **Removing visualization code** (no matplotlib/cartopy dependencies)
- **Replacing PNG outputs** with NetCDF serialization
- **Adding explicit normalization** with mean/std tracking
- **Implementing data validation** and quality checks
- **Supporting batch processing** for multiple cycles and forecast hours

## Key Files

### Core Implementation
- **`tools/create_training_pipeline.py`**: Main training pipeline implementation
- **`test_pipeline_mock.py`**: Mock tests for core functionality validation
- **`example_usage.py`**: Usage examples and demonstration scripts

### Existing Infrastructure (Reused)
- **`hrrr_processor_refactored.py`**: GRIB data loading and processing logic
- **`field_registry.py`**: Field configuration management
- **`derived_params/`**: Meteorological calculations and derived parameters
- **`parameters/*.json`**: Variable definitions and configurations

## Architecture

The training pipeline follows this data flow:

```
HRRR GRIB Files → Field Loading → Derived Calculations → Validation → Normalization → NetCDF Serialization
```

### Key Components

1. **HRRRTrainingPipeline**: Main pipeline class
2. **Variable Selection**: Flexible filtering by category, derived status, etc.
3. **Data Loading**: Reuses existing HRRRProcessor logic without visualization
4. **Quality Validation**: Checks for NaN values, zero variance, reasonable ranges
5. **Normalization**: Z-score normalization with mean/std stored as attributes
6. **NetCDF Output**: Compressed, structured datasets with proper metadata

## Usage

### Basic Python Usage

```python
from tools.create_training_pipeline import HRRRTrainingPipeline

# Initialize pipeline
pipeline = HRRRTrainingPipeline()

# Select variables
variables = pipeline.select_training_variables(
    include_categories=['severe', 'instability', 'surface'],
    include_derived=True,
    max_variables=20
)

# Generate training data
summary = pipeline.generate_training_data(
    cycles=['2025060900', '2025060912'],
    forecast_hours=[0, 1, 2, 3],
    output_base_dir=Path('./training_data'),
    variables=variables
)
```

### Command Line Usage

```bash
# Generate training data for multiple cycles
python tools/create_training_pipeline.py \
    --cycles 2025060900 2025060912 \
    --forecast-hours 0 1 2 3 \
    --output-dir ./training_data \
    --categories severe instability surface \
    --max-variables 25

# Quick test with limited scope
python tools/create_training_pipeline.py \
    --cycles 2025060900 \
    --forecast-hours 0 \
    --output-dir ./quick_test \
    --max-variables 10
```

## Variable Selection

The pipeline supports flexible variable selection:

### By Category
- `severe`: Severe weather parameters (SCP, STP, etc.)
- `instability`: Atmospheric instability indices
- `surface`: Surface meteorological fields
- `atmospheric`: Column and boundary layer parameters
- `reflectivity`: Radar reflectivity fields
- `smoke`: Smoke concentration and dispersion

### By Type
- **Direct GRIB fields**: Temperature, wind, pressure, etc.
- **Derived parameters**: Composite indices, calculated from multiple inputs

### Filtering Options
- `include_categories`: Specify categories to include
- `exclude_categories`: Specify categories to exclude
- `include_derived`: Whether to include calculated parameters
- `max_variables`: Limit total number of variables

## Output Format

### NetCDF Structure
Each output file contains:
- **Data variables**: Normalized meteorological fields (float32)
- **Coordinates**: Latitude/longitude grids
- **Global attributes**: Metadata about cycle, forecast hour, processing
- **Variable attributes**: Normalization parameters, units, descriptions

### File Organization
```
output_directory/
├── cycle_2025060900/
│   ├── forecast_hour_F00.nc
│   ├── forecast_hour_F01.nc
│   └── ...
├── cycle_2025060912/
│   ├── forecast_hour_F00.nc
│   └── ...
└── generation_summary.json
```

### Normalization Attributes
Each variable includes:
- `normalization_mean`: Original data mean
- `normalization_std`: Original data standard deviation  
- `original_min`: Original data minimum
- `original_max`: Original data maximum
- `valid_points`: Number of valid (non-NaN) points

## Data Quality

### Validation Checks
1. **NaN Detection**: Identifies all-NaN or excessive NaN values
2. **Zero Variance**: Detects constant fields (std < 1e-10)
3. **Range Validation**: Checks for reasonable meteorological values
4. **Coordinate Consistency**: Ensures all fields have matching grids

### Error Handling
- Failed field loading is logged but doesn't stop processing
- Quality validation failures exclude fields from output
- GRIB download failures are retried across multiple sources
- Processing continues with available data

## Testing

### Mock Tests
Run core functionality tests without requiring GRIB downloads:
```bash
python test_pipeline_mock.py
```

### Full Tests  
Test with real GRIB data (requires network access):
```bash
python test_training_pipeline.py
```

### Example Usage
Explore pipeline capabilities:
```bash
python example_usage.py
```

## Performance Considerations

### Memory Management
- Datasets are explicitly closed after saving
- GRIB files are cached locally to avoid re-downloading
- Float32 precision used for storage efficiency

### Processing Time
- Derived parameters add computational overhead
- Smoke fields are memory-intensive (consider excluding)
- Batch processing allows monitoring progress
- Timeouts prevent hanging on problematic fields

### Storage Requirements
Approximate file sizes:
- **10 variables**: ~0.5 MB per forecast hour
- **25 variables**: ~1.2 MB per forecast hour  
- **50 variables**: ~2.5 MB per forecast hour

## Integration with Diffusion Models

### Dataset Loading
```python
import xarray as xr

# Load training dataset
ds = xr.open_dataset('cycle_2025060900/forecast_hour_F00.nc')

# Access normalized data
temperature = ds['t2m'].values  # Already normalized (mean=0, std=1)

# Get original statistics for denormalization
orig_mean = ds['t2m'].attrs['normalization_mean']
orig_std = ds['t2m'].attrs['normalization_std']

# Denormalize if needed
original_temp = (temperature * orig_std) + orig_mean
```

### Training Considerations
1. **Normalization**: Data is pre-normalized (mean=0, std=1)
2. **Missing Values**: NaN values should be handled in model architecture  
3. **Spatial Structure**: Data maintains original HRRR grid structure
4. **Temporal Sequences**: Multiple forecast hours can create time series
5. **Variable Selection**: Choose variables relevant to target phenomena

## Customization

### Adding New Variables
1. Add configuration to appropriate `parameters/*.json` file
2. Implement calculation function in `derived_params/` if derived
3. Register function in `derived_params/__init__.py`
4. Pipeline will automatically detect and process new variables

### Custom Processing
Extend `HRRRTrainingPipeline` class:
```python
class CustomPipeline(HRRRTrainingPipeline):
    def custom_processing_step(self, data):
        # Add custom data processing
        return processed_data
```

## Troubleshooting

### Common Issues

1. **GRIB Download Failures**
   - Check network connectivity
   - Verify cycle dates are not too recent
   - Try older cycles (data availability varies)

2. **Memory Issues**
   - Reduce `max_variables` parameter
   - Exclude memory-intensive categories (smoke, updraft_helicity)
   - Process smaller time ranges

3. **NetCDF Errors**
   - Ensure output directory is writable
   - Check for disk space
   - Verify xarray/netCDF4 installation

4. **Validation Failures**
   - Check GRIB file integrity
   - Verify field configurations are correct
   - Review derived parameter inputs

### Debugging
Enable verbose logging by modifying print statements in pipeline code or add logging configuration.

## Future Enhancements

### Potential Improvements
1. **Parallel Processing**: Multi-threaded field loading
2. **Cloud Storage**: Direct output to cloud storage systems
3. **Data Augmentation**: Spatial/temporal augmentation techniques
4. **Format Support**: Additional output formats (HDF5, Zarr)
5. **Quality Metrics**: Advanced data quality assessment
6. **Metadata Standards**: Enhanced CF-compliant metadata

### Machine Learning Integration
1. **PyTorch DataLoaders**: Direct integration with PyTorch training
2. **TensorFlow Dataset**: TensorFlow data pipeline integration  
3. **Dask Support**: Large-scale distributed processing
4. **MLflow Integration**: Experiment tracking and model versioning

---

## Summary

The HRRR Training Pipeline successfully transforms the existing visualization-focused system into a robust numerical data processing pipeline suitable for ML training. It maintains all the sophisticated meteorological calculations while replacing visualization outputs with properly structured, normalized NetCDF datasets.

Key achievements:
- ✅ **Removed visualization dependencies** - No matplotlib/cartopy required
- ✅ **Implemented NetCDF serialization** - Structured, compressed output
- ✅ **Added explicit normalization** - Mean/std tracked as metadata
- ✅ **Included data validation** - Quality checks and error handling
- ✅ **Supported batch processing** - Multiple cycles and forecast hours
- ✅ **Maintained extensibility** - Leverages existing field registry system
- ✅ **Provided comprehensive testing** - Mock tests and usage examples

This pipeline is ready for production use in diffusion model training workflows.