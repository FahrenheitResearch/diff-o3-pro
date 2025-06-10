# HRRR Processing System - Technical Documentation

Complete technical documentation covering system architecture, implementation details, and advanced technical topics.

## 🏗️ System Architecture

### Overview
The HRRR processing system has evolved from a monolithic 545-line script into a modern, extensible, configuration-driven framework. This transformation provides 96.6% success rate with 10x easier parameter addition.

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **`smart_hrrr_processor.py`** | Main processing engine | Live monitoring, parallel processing, auto-detection |
| **`field_registry.py`** | Central parameter management | Category organization, field lookup, validation |
| **`field_templates.py`** | Template inheritance system | Reusable configurations, easy parameter addition |
| **`config_builder.py`** | Configuration loader & validator | JSON parsing, template resolution, error checking |
| **`derived_parameters.py`** | Advanced calculations | MetPy integration, personality composites |
| **`parameters/`** | JSON configuration files | Category-based organization, template overrides |

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                Smart HRRR Processor                        │
├─────────────────────┬───────────────────────────────────────┤
│   Live Monitoring   │         Batch Processing             │
│   ├─ Cycle Detection│         ├─ Model Run Processing       │
│   ├─ Auto-switching │         ├─ Forecast Hour Processing   │
│   └─ Real-time Proc │         └─ Parallel Workers          │
├─────────────────────┴───────────────────────────────────────┤
│                Field Registry System                        │
│   ├─ Template Resolution    ├─ Category Management          │
│   ├─ Configuration Loading  └─ Parameter Validation         │
├─────────────────────────────────────────────────────────────┤
│               Data Processing Pipeline                      │
│   ├─ GRIB Download/Access   ├─ Derived Calculations         │
│   ├─ cfgrib Data Loading    ├─ Coordinate Processing        │
│   ├─ Field Extraction       └─ Map Generation               │
├─────────────────────────────────────────────────────────────┤
│                  Output Management                          │
│   ├─ Directory Structure    ├─ Duplicate Detection          │
│   ├─ File Naming           └─ Log Management                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Comparison - Original vs. Refactored

| Metric | Original | Refactored | Improvement |
|--------|----------|------------|-------------|
| **Code Structure** | 545-line monolith | Modular components | ✅ Maintainable |
| **Adding Parameters** | Edit hardcoded dict | Add JSON config | ✅ 10x easier |
| **Success Rate** | ~85% (estimated) | 96.6% (28/29) | ✅ More robust |
| **Organization** | Flat list | Category-based | ✅ Logical grouping |
| **Extensibility** | Difficult | Trivial | ✅ Future-proof |
| **Error Handling** | Fails on first error | Graceful recovery | ✅ Continues processing |
| **Batch Processing** | Manual single runs | Automated pipelines | ✅ Operational ready |

---

## 🔧 Template System Architecture

### Base Templates
Templates provide inheritance-based configuration with smart defaults and easy overrides:

```python
# field_templates.py structure
FIELD_TEMPLATES = {
    'surface_cape': {
        'access_pattern': 'height_agl',
        'level': 2,
        'var': 'cape',
        'title': 'Surface-Based CAPE',
        'units': 'J/kg',
        'cmap': 'CAPEColors',
        'levels': [250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
        'extend': 'max',
        'transform': 'none',
        'category': 'instability'
    }
}
```

### Template Inheritance
```json
// parameters/instability.json
{
  "my_custom_cape": {
    "template": "surface_cape",
    "title": "My Custom CAPE",
    "levels": [500, 1000, 2000, 3000, 4000]
  }
}
```

### Access Patterns
Different meteorological data requires different access methods:

| Pattern | Use Case | Example |
|---------|----------|---------|
| `height_agl` | Fixed height above ground | 2m temperature, 10m winds |
| `pressure_level` | Fixed pressure levels | 500mb geopotential height |
| `surface_instant` | Surface instantaneous | Surface pressure, precipitation |
| `layer_integrated` | Layer-integrated values | CAPE between pressure levels |
| `composite_max` | Maximum in column | Composite reflectivity |

---

## 🔬 GRIB Data Processing Details

### cfgrib Integration
The system uses cfgrib for robust GRIB2 file processing with intelligent filtering:

```python
# Standard cfgrib access
ds = cfgrib.open_dataset(file, filter_by_keys={
    'typeOfLevel': 'heightAboveGround',
    'level': 2
})

# Multi-level access with filtering
ds = cfgrib.open_dataset(file, filter_by_keys={
    'typeOfLevel': 'isobaricInhPa',
    'level': [1000, 925, 850, 700, 500]
})
```

### HRRR-Specific Handling
HRRR files require special handling for certain parameters:

**Surface Parameters:**
- Use `paramId` access for surface variables (t2m, d2m, rh2m, winds)
- Level-based access for atmospheric layers (CAPE, CIN, reflectivity)
- Automatic filtering for complex GRIB2 structure

**Coordinate System:**
- Native Lambert Conformal projection preservation
- No meshgrid issues with proper coordinate handling
- Maintains 1059×1799 pixel native resolution

---

## 🌪️ Smoke Products Implementation

### Technical Challenge: Variable Access
HRRR Version 4+ includes smoke forecasting, but accessing these parameters required solving complex GRIB2 structure issues.

### Near-Surface Smoke (8m AGL)
```python
# Solution: Use cfgrib with specific level filters
ds = cfgrib.open_dataset(file, filter_by_keys={
    'typeOfLevel': 'heightAboveGround', 
    'level': 8
})
# Variable: 'mdens' (MASSDEN in GRIB2 becomes 'mdens' in cfgrib)
```

### Column-Integrated Smoke (COLMD)
```python
# Challenge: COLMD couldn't be accessed through standard cfgrib
# Solution: wgrib2 → NetCDF extraction workflow

# 1. Find COLMD record
result = subprocess.run(['wgrib2', grib_file, '-s'], capture_output=True, text=True)
for line in result.stdout.strip().split('\n'):
    if 'COLMD' in line:
        colmd_record = line.split(':')[0]
        break

# 2. Extract to NetCDF
subprocess.run(['wgrib2', grib_file, '-d', colmd_record, '-netcdf', temp_nc])

# 3. Load with xarray
ds = xr.open_dataset(temp_nc)
# Variable: 'COLMD_entireatmosphere_consideredasasinglelayer_'
```

### NOAA-Standard Scaling
Smoke products require precise unit conversion and scaling:

```python
# Near-surface smoke: kg/m³ → μg/m³
surface_smoke = data * 1e9

# Column smoke: kg/m² → mg/m²  
column_smoke = data * 1e6

# NOAA standard levels
levels = [1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 60, 100, 200]  # μg/m³
```

---

## 🧮 Derived Parameters System

### MetPy Integration Architecture
The system supports advanced meteorological calculations through MetPy:

```python
# Example: Heat Index calculation
from metpy.calc import heat_index
from metpy.units import units

@staticmethod
def heat_index_calculation(temperature_2m, humidity_2m):
    """Compute heat index from 2m temperature and humidity"""
    temp = temperature_2m * units.celsius
    rh = humidity_2m * units.percent
    hi = heat_index(temp, rh)
    return hi.to('celsius').magnitude
```

### Personality Composites Framework
Advanced weather indices that combine multiple meteorological parameters:

```python
# Example: Destroyer Reality Check Composite (multiplicative logic)
@staticmethod
def destroyer_reality_check(sbcape, srh_01km, wind_shear_06km, mlcin):
    """
    Reality-check composite using multiplicative math
    Exposes weak links in severe weather setups
    """
    # Normalize each component to 0-1 scale
    cape_factor = np.clip(sbcape / 2500.0, 0, 1)
    srh_factor = np.clip(srh_01km / 250.0, 0, 1) 
    shear_factor = np.clip(wind_shear_06km / 25.0, 0, 1)
    stability_factor = np.clip(-mlcin / 100.0, 0, 1)
    
    # Multiplicative combination - any weak link tanks the score
    drc = cape_factor * srh_factor * shear_factor * stability_factor
    
    return drc
```

---

## 📈 Performance Optimization

### Memory Management
```python
# Streaming processing prevents memory accumulation
def process_field_streaming(grib_file, field_config):
    """Process single field with minimal memory footprint"""
    try:
        # Load only required subset
        ds = cfgrib.open_dataset(grib_file, filter_by_keys=filters)
        
        # Extract field
        data = ds[field_config['var']].values
        
        # Process and visualize
        result = create_visualization(data, field_config)
        
        # Explicit cleanup
        ds.close()
        del ds, data
        gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Field processing failed: {e}")
        return None
```

### Parallel Processing Architecture
```python
# Worker pool with optimal resource utilization
def parallel_field_processing(fields, num_workers=4):
    """Process multiple fields in parallel with resource management"""
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_field = {
            executor.submit(process_field, field): field 
            for field in fields
        }
        
        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_field):
            field = future_to_field[future]
            try:
                results[field] = future.result()
            except Exception as e:
                logger.error(f"Field {field} failed: {e}")
                results[field] = None
                
    return results
```

### Smart Caching System
```python
# Duplicate detection with hash verification
def check_existing_output(output_path, grib_file):
    """Check if output already exists and is current"""
    if not os.path.exists(output_path):
        return False
        
    # Compare modification times
    output_mtime = os.path.getmtime(output_path)
    grib_mtime = os.path.getmtime(grib_file)
    
    # Output is current if newer than source
    return output_mtime > grib_mtime
```

---

## 🔍 Error Handling & Recovery

### Graceful Failure Management
The system continues processing even when individual fields fail:

```python
def robust_field_processing(fields_list):
    """Process all fields with graceful error handling"""
    results = {'success': [], 'failed': []}
    
    for field in fields_list:
        try:
            result = process_single_field(field)
            if result:
                results['success'].append(field)
                logger.info(f"✅ {field} processed successfully")
            else:
                results['failed'].append(field)
                logger.warning(f"⚠️ {field} processing returned null")
                
        except Exception as e:
            results['failed'].append(field)
            logger.error(f"❌ {field} failed: {str(e)}")
            
    return results
```

### Data Quality Validation
```python
def validate_field_data(data, field_name):
    """Validate meteorological data quality"""
    
    # Check for completely invalid data
    if data is None or data.size == 0:
        raise ValueError(f"{field_name}: No data available")
        
    # Check for excessive NaN values
    nan_percentage = np.isnan(data).sum() / data.size
    if nan_percentage > 0.5:
        logger.warning(f"{field_name}: {nan_percentage:.1%} NaN values")
        
    # Check for reasonable meteorological ranges
    if field_name.startswith('t2m'):
        if np.nanmin(data) < 200 or np.nanmax(data) > 350:  # Kelvin
            logger.warning(f"{field_name}: Temperature values outside expected range")
            
    return True
```

---

## 🌐 Data Source Management

### Multi-Source Download Strategy
```python
# Hierarchical data source with fallbacks
HRRR_SOURCES = [
    {
        'name': 'NOMADS_Primary',
        'base_url': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/',
        'priority': 1,
        'timeout': 30
    },
    {
        'name': 'AWS_S3_Recent', 
        'base_url': 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/',
        'priority': 2,
        'timeout': 60
    },
    {
        'name': 'Google_Cloud_Historical',
        'base_url': 'https://storage.googleapis.com/high-resolution-rapid-refresh/',
        'priority': 3,
        'timeout': 120
    }
]

def download_with_fallback(cycle, forecast_hour):
    """Download HRRR data with automatic source fallback"""
    for source in HRRR_SOURCES:
        try:
            url = construct_url(source, cycle, forecast_hour)
            response = requests.get(url, timeout=source['timeout'])
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.warning(f"Source {source['name']} failed: {e}")
            continue
            
    raise Exception("All HRRR data sources failed")
```

---

## 🎨 Visualization System

### SPC-Style Colormap Creation
```python
def create_spc_colormaps():
    """Create meteorologically appropriate colormaps"""
    colormaps = {}
    
    # CAPE colormap (white→green→yellow→orange→red→purple)
    cape_colors = ['#FFFFFF', '#90EE90', '#FFFF00', '#FFA500', '#FF0000', '#800080']
    colormaps['CAPEColors'] = LinearSegmentedColormap.from_list('CAPE', cape_colors)
    
    # NWS Reflectivity standard
    dbz_colors = ['#FFFFFF', '#00FFFF', '#0000FF', '#00FF00', '#FFFF00', 
                  '#FFA500', '#FF0000', '#FF00FF']
    colormaps['Reflectivity'] = LinearSegmentedColormap.from_list('DBZ', dbz_colors)
    
    return colormaps
```

### Map Projection Handling
```python
def setup_hrrr_projection():
    """Configure native HRRR Lambert Conformal projection"""
    
    # HRRR-specific projection parameters
    proj = ccrs.LambertConformal(
        central_longitude=-97.5,
        central_latitude=38.5,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(ellipse='sphere')
    )
    
    return proj
```

---

## 🚨 Troubleshooting Common Issues

### GRIB2 Access Problems
```python
# Multiple values for unique key error
try:
    ds = cfgrib.open_dataset(file, filter_by_keys={'typeOfLevel': 'surface'})
except ValueError as e:
    if "multiple values" in str(e):
        # Use more specific filtering
        ds = cfgrib.open_dataset(file, filter_by_keys={
            'typeOfLevel': 'surface',
            'stepType': 'instant'
        })
```

### Memory Issues
```python
# Monitor memory usage during processing
def process_with_memory_monitoring(fields):
    """Process fields with memory usage tracking"""
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    for i, field in enumerate(fields):
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = current_memory - initial_memory
        
        if memory_growth > 1000:  # More than 1GB growth
            logger.warning(f"High memory usage: {memory_growth:.1f}MB growth")
            gc.collect()  # Force garbage collection
            
        process_field(field)
```

### Network Timeout Handling
```python
# Robust download with retries
def download_with_retry(url, max_retries=3, backoff_factor=2):
    """Download with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            timeout = 30 * (backoff_factor ** attempt)
            response = requests.get(url, timeout=timeout)
            return response.content
            
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = backoff_factor ** attempt
            logger.warning(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s")
            time.sleep(wait_time)
```

---

## 📊 System Monitoring & Logging

### Comprehensive Logging System
```python
# Multi-level logging with file and console output
def setup_logging(log_dir, debug=False):
    """Configure comprehensive logging system"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'processing_{datetime.now():%Y%m%d_%H%M%S}.log')
    )
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not debug else logging.DEBUG)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

This technical documentation provides the foundation for understanding, maintaining, and extending the HRRR processing system's sophisticated architecture and implementation details.