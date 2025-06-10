# HRRR Diffusion Model Training Data

## Overview
This document describes a comprehensive training dataset derived from NOAA's High-Resolution Rapid Refresh (HRRR) meteorological model, specifically prepared for diffusion model training for weather prediction and analysis.

## Dataset Specifications

### Spatial Resolution
- **Native Grid**: 3km resolution covering continental United States (CONUS)
- **Grid Dimensions**: 1059 × 1799 grid points
- **Total Points**: 1,905,141 spatial locations per variable
- **Coordinate System**: Lambert Conformal Conic projection
- **Coverage**: Approximately 21°N to 48°N latitude, 134°W to 60°W longitude

### Temporal Coverage
- **Date Range**: April 2025 (30 days)
- **Model Runs**: 00Z, 06Z, 12Z, 18Z (4 runs per day)
- **Forecast Hours**: F00-F48 (49 forecast hours per run)
- **Total Files**: 5,880 NetCDF files
- **Temporal Resolution**: Hourly forecasts

### Data Format
- **File Format**: NetCDF4 (.nc)
- **Compression**: zlib level 4
- **Data Type**: 32-bit floating point
- **Normalization**: Z-score normalized (mean=0, std=1)
- **File Size**: ~72 MB per file
- **Total Dataset Size**: ~420 GB

## Meteorological Variables (17 total)

### Surface Variables (5)
| Variable | Description | Units | Physical Meaning |
|----------|-------------|-------|------------------|
| `t2m` | 2-meter Temperature | K | Near-surface air temperature |
| `d2m` | 2-meter Dewpoint | K | Near-surface moisture content |
| `u10` | 10-meter U-wind | m/s | East-west wind component |
| `v10` | 10-meter V-wind | m/s | North-south wind component |
| `sp` | Surface Pressure | Pa | Atmospheric pressure at surface |

### Upper-Air Variables (7)
| Variable | Description | Level | Units | Physical Meaning |
|----------|-------------|-------|-------|------------------|
| `temp_850` | Temperature | 850 mb | K | Low-level atmospheric temperature |
| `temp_700` | Temperature | 700 mb | K | Mid-level atmospheric temperature |
| `temp_500` | Temperature | 500 mb | K | Mid-level atmospheric temperature |
| `dewpoint_850` | Dewpoint Temperature | 850 mb | K | Low-level moisture content |
| `rh_500` | Relative Humidity | 500 mb | % | Mid-level moisture saturation |
| `u_500` | U-wind Component | 500 mb | m/s | Mid-level east-west wind |
| `v_500` | V-wind Component | 500 mb | m/s | Mid-level north-south wind |

### Instability Variables (3)
| Variable | Description | Units | Physical Meaning |
|----------|-------------|-------|------------------|
| `sbcape` | Surface-Based CAPE | J/kg | Convective instability energy |
| `mlcape` | Mixed-Layer CAPE | J/kg | Mixed-layer convective energy |
| `sbcin` | Surface-Based CIN | J/kg | Convective inhibition energy |

### Reflectivity Variables (2)
| Variable | Description | Level | Units | Physical Meaning |
|----------|-------------|-------|-------|------------------|
| `reflectivity_1km` | Radar Reflectivity | 1 km AGL | dBZ | Low-level precipitation intensity |
| `reflectivity_4km` | Radar Reflectivity | 4 km AGL | dBZ | Mid-level precipitation intensity |

## Data Normalization

All variables are z-score normalized:
```python
normalized_value = (raw_value - mean) / std_deviation
```

Normalization statistics are saved alongside each NetCDF file in accompanying JSON files for denormalization during inference.

## File Naming Convention
```
HRRR_TIER2_YYYYMMDDHH_FXX.nc
```
- `YYYY`: Year (2025)
- `MM`: Month (04)
- `DD`: Day (01-30)
- `HH`: Model run hour (00, 06, 12, 18)
- `XX`: Forecast hour (00-48)

Example: `HRRR_TIER2_2025040100_F06.nc` = April 1, 2025, 00Z run, 6-hour forecast

## Data Access Patterns

### Single File Structure
```python
import xarray as xr

ds = xr.open_dataset('HRRR_TIER2_2025040100_F00.nc')
print(ds.data_vars)  # Lists all 17 variables
print(ds.dims)       # {'y': 1059, 'x': 1799}

# Access specific variable
temperature = ds['t2m']  # Shape: (1059, 1799)
```

### Batch Loading
```python
# Load multiple files for time series
files = glob.glob('HRRR_TIER2_20250401*_F00.nc')  # All F00 for April 1
ds_combined = xr.open_mfdataset(files, combine='by_coords')
```

## Recommended Use Cases

### 1. Weather Forecasting Diffusion Models
- **Input**: Current atmospheric state (17 variables)
- **Target**: Future atmospheric state at various lead times
- **Architecture**: U-Net, Vision Transformer, or custom CNN-based diffusion models

### 2. Nowcasting Applications
- **Focus**: Short-term precipitation and severe weather prediction
- **Key Variables**: Reflectivity, CAPE, surface conditions
- **Temporal Range**: 0-6 hour forecasts

### 3. Climate Pattern Analysis
- **Application**: Long-term atmospheric pattern learning
- **Approach**: Multi-variate time series analysis
- **Temporal Range**: Full F00-F48 forecast evolution

### 4. Downscaling Research
- **Purpose**: Learn relationships between different atmospheric scales
- **Method**: Train on subset of variables, predict others
- **Focus**: Temperature-moisture-wind relationships

## Model Architecture Considerations

### Spatial Dimensions
- **Input Shape**: `(batch, 17, 1059, 1799)` for all variables
- **Memory Requirements**: ~500 MB per batch item (FP32)
- **Recommendation**: Use mixed precision (FP16) training

### Temporal Modeling Options
1. **Single Time Step**: Predict next state from current state
2. **Multi-Step**: Predict multiple future time steps simultaneously
3. **Forecast Sequence**: Use multiple forecast hours as sequence data

### Variable Relationships
- **Physical Constraints**: Geostrophic balance, hydrostatic equilibrium
- **Correlation Structure**: Temperature-dewpoint coupling, wind-pressure relationships
- **Multi-level Consistency**: Ensure vertical atmospheric profile coherence

## Data Quality and Limitations

### Strengths
- ✅ Native 3km resolution (high spatial detail)
- ✅ Comprehensive variable selection for severe weather
- ✅ Consistent normalization and formatting
- ✅ High temporal resolution (hourly)
- ✅ Real operational model data quality

### Limitations
- ⚠️ Single month (April 2025) - limited seasonal coverage
- ⚠️ CONUS domain only - no global coverage
- ⚠️ Some specialized variables missing (wind shear, updraft helicity)
- ⚠️ Static fields (orography) not included

### Missing Variables (for reference)
- `orog`: Orography/terrain height (static field)
- `reflectivity_comp`: Composite reflectivity (encoding issues)
- Various derived severe weather indices (STP, SCP, etc.)

## Recommended Training Strategies

### 1. Progressive Training
Start with subset of variables, gradually add complexity:
1. Surface variables only (5 variables)
2. Add upper-air (12 variables)  
3. Full dataset (17 variables)

### 2. Multi-Task Learning
- Primary task: Weather prediction
- Auxiliary tasks: Variable consistency, physical constraint satisfaction

### 3. Temporal Sampling Strategies
- **Dense sampling**: F00, F01, F02, F03... (for short-term accuracy)
- **Sparse sampling**: F00, F06, F12, F18... (for efficiency)
- **Lead-time specific**: Different models for different forecast horizons

### 4. Data Augmentation
- Spatial cropping and patching
- Temporal sequence permutation
- Variable masking for robust learning

## Performance Benchmarks

### Expected Model Performance Targets
- **Temperature**: RMSE < 2K at F06, < 3K at F24
- **Precipitation**: CSI > 0.3 for >1mm/hr at F06
- **Wind**: RMSE < 3 m/s at F12
- **CAPE**: Correlation > 0.8 with observations at F06

### Computational Requirements
- **Training**: 4-8 GPU setup recommended for full spatial resolution
- **Inference**: Single GPU capable for operational deployment
- **Memory**: 32GB+ VRAM for full resolution training

## Usage Examples

### Basic Data Loading
```python
import xarray as xr
import numpy as np

# Load single file
ds = xr.open_dataset('HRRR_TIER2_2025040100_F00.nc')

# Get all surface variables
surface_vars = ['t2m', 'd2m', 'u10', 'v10', 'sp']
surface_data = ds[surface_vars].to_array().values  # Shape: (5, 1059, 1799)

# Load normalization stats
import json
with open('HRRR_TIER2_2025040100_F00.json', 'r') as f:
    norm_stats = json.load(f)
```

### Training Data Generator
```python
class HRRRDataset:
    def __init__(self, file_pattern, variables=None):
        self.files = glob.glob(file_pattern)
        self.variables = variables or ['t2m', 'd2m', 'u10', 'v10', 'sp']
    
    def __getitem__(self, idx):
        ds = xr.open_dataset(self.files[idx])
        data = ds[self.variables].to_array().values
        return torch.tensor(data, dtype=torch.float32)
```

## Contact and Support

This dataset was generated using the HRRR Tier 2 Enhanced Severe Weather Training Pipeline. For questions about data format, variable definitions, or processing methodology, refer to the accompanying pipeline documentation.

**Data Source**: NOAA High-Resolution Rapid Refresh (HRRR) Model
**Processing Pipeline**: Custom Python/xarray/cfgrib workflow
**Format Standard**: CF-compliant NetCDF4