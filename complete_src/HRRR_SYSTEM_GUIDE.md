# HRRR Weather Map System Guide

*A comprehensive guide for understanding and extending the HRRR weather visualization system*

## System Overview

This system processes High-Resolution Rapid Refresh (HRRR) weather model data to create professional weather maps in the style of NOAA's Storm Prediction Center. The architecture is configuration-driven, making it easy to add new parameters and visualization styles without modifying core code.

## Architecture Components

### Core Files Structure
```
hrrr_processor_refactored.py    # Main plotting engine
smart_hrrr_processor.py         # Orchestration & workflows  
field_registry.py               # Configuration management
field_templates.py              # Template system
parameters/                     # JSON field definitions
│   ├── instability.json       # CAPE, CIN, etc.
│   ├── severe.json            # Severe weather parameters
│   ├── surface.json           # Surface observations
│   ├── derived.json           # Calculated composites
│   └── ...                    # More categories
derived_params/                 # Custom calculations
│   ├── __init__.py           # Parameter dispatch
│   ├── common.py             # Shared utilities
│   └── *.py                  # Individual calculations
```

### Data Flow
```
NOMADS GRIB → Field Registry → Template Resolution → Data Loading → Processing → Maps
```

## Understanding Field Configurations

### Basic Field Structure
Every meteorological parameter is defined by a JSON configuration:

```json
{
  "field_name": {
    "title": "Human-readable name",
    "units": "Physical units", 
    "cmap": "colormap_name",
    "levels": [0, 10, 20, 30, 40],
    "extend": "max",
    "category": "severe",
    "access_pattern": "surface_instant",
    "var": "grib_variable_name"
  }
}
```

### Key Configuration Properties

**Data Access**:
- `access_pattern`: How to extract from GRIB (`surface_instant`, `height_layer`, `param_id`)
- `var`: GRIB variable name
- `param_id`: GRIB parameter ID (alternative to var)
- `level`: Height/pressure level for atmospheric data

**Visualization**:
- `cmap`: Colormap name (from built-in collection)
- `levels`: Contour/fill levels
- `extend`: Handle values outside level range (`min`, `max`, `both`)
- `plot_style`: Visualization method (`filled`, `lines`, `composite`)

**Organization**:
- `category`: Logical grouping for batch processing
- `title`: Display name on maps
- `units`: Physical units for labeling

### Template System

Templates provide inheritance for common field types:

```json
{
  "surface_temperature": {
    "template": "temperature_base",
    "level": 2,
    "title": "2m Temperature"
  }
}
```

Templates are defined in `field_templates.py` and include styling, units, and access patterns.

### Access Patterns

**`surface_instant`**: Surface-level instantaneous fields
```json
{
  "access_pattern": "surface_instant",
  "var": "t2m"
}
```

**`height_layer`**: Layer between two heights
```json
{
  "access_pattern": "height_layer", 
  "var": "hlcy",
  "level": [1000, 0]  # 0-1km layer
}
```

**`param_id`**: Direct parameter ID access
```json
{
  "access_pattern": "param_id",
  "param_id": 260242,
  "var": "cape"
}
```

## Plot Styles and Visualization

### Available Plot Styles

**`filled` (default)**: Standard filled contours
- Creates colored regions between contour levels
- Best for continuous fields like temperature, humidity

**`lines`**: Contour lines only  
- Clean line contours without fill
- Good for overlays and parameter boundaries

**`multicolor_lines`**: Progressive color contours
- Different colored lines for different ranges
- SPC-style progression from green → yellow → red

**`composite`**: Base field + overlay
- Filled base layer with contour overlay
- Example: MLCIN shading + VTP contours

**`spc_vtp`**: Specialized VTP panel
- MLCIN shading with hatching
- Dashed CIN isolines 
- VTP contour overlay

### Custom Plot Styles

Add new plot styles by extending the plotting logic in `hrrr_processor_refactored.py`:

```python
elif plot_style == 'my_custom_style':
    # Custom plotting logic here
    cs = ax_map.contour(lons, lats, plot_data,
                       levels=field_config['levels'],
                       colors='custom_color',
                       transform=ccrs.PlateCarree())
```

### Colormap System

The system includes meteorological colormaps optimized for weather visualization:

**NWS Standard**:
- `NWSReflectivity`: Radar reflectivity colors
- `CAPE`: Yellow to red for instability
- `CIN`: Blues for convective inhibition

**Severe Weather**:
- `STP`: Significant Tornado Parameter
- `SCP`: Supercell Composite Parameter  
- `EHI`: Energy Helicity Index

**Custom**: Create new colormaps in `create_spc_colormaps()`:

```python
my_colors = ['#color1', '#color2', '#color3']
colormaps['MyColormap'] = LinearSegmentedColormap.from_list('MyMap', my_colors)
```

## Adding New Map Products

### Method 1: Direct GRIB Fields

For parameters directly available in HRRR GRIB files:

1. **Identify the GRIB variable**: Use `grib_ls` or check HRRR documentation
2. **Add to appropriate JSON file**:

```json
{
  "my_new_field": {
    "access_pattern": "surface_instant",
    "var": "new_var",
    "title": "My New Parameter", 
    "units": "unit",
    "cmap": "viridis",
    "levels": [0, 5, 10, 15, 20],
    "extend": "max",
    "category": "surface"
  }
}
```

3. **Test**: Run `python smart_hrrr_processor.py --latest --fields my_new_field`

### Method 2: Derived Parameters

For calculated parameters combining multiple GRIB fields:

1. **Create calculation function** in `derived_params/my_parameter.py`:

```python
import numpy as np
from .common import *

def my_parameter(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    """
    Calculate my custom parameter.
    
    Args:
        field1: First input field
        field2: Second input field
        
    Returns:
        Calculated parameter values
    """
    result = np.sqrt(field1**2 + field2**2)  # Example calculation
    return result
```

2. **Register in dispatch table** (`derived_params/__init__.py`):

```python
DERIVED_PARAMETER_FUNCTIONS = {
    # ... existing functions ...
    'my_parameter': my_parameter,
}
```

3. **Add configuration** to `parameters/derived.json`:

```json
{
  "my_parameter": {
    "title": "My Custom Parameter",
    "units": "custom_unit",
    "cmap": "plasma", 
    "levels": [0, 1, 2, 4, 8, 16],
    "extend": "max",
    "category": "derived",
    "derived": true,
    "inputs": ["field1", "field2"],
    "function": "my_parameter"
  }
}
```

### Method 3: Composite Maps

For maps combining multiple parameters with different visualization styles:

1. **Use existing composite system**:

```json
{
  "my_composite": {
    "title": "My Composite Map",
    "plot_style": "composite", 
    "base_field": "base_parameter",
    "overlay_field": "overlay_parameter",
    "overlay_colors": ["red"],
    "overlay_linewidths": [2.0],
    "category": "composite"
  }
}
```

## Advanced Features

### Templates and Inheritance

Create reusable templates for similar parameters:

```python
# In field_templates.py
TEMPLATES = {
    'wind_component': {
        'units': 'm/s',
        'cmap': 'RdBu_r', 
        'levels': [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        'extend': 'both',
        'category': 'surface'
    }
}
```

Use templates:

```json
{
  "u_wind_850": {
    "template": "wind_component",
    "access_pattern": "pressure_level",
    "var": "u", 
    "level": 850,
    "title": "850mb U-Wind"
  }
}
```

### Workflows and Automation

Create custom workflows in `custom_workflows.json`:

```json
{
  "severe_weather_suite": {
    "description": "Complete severe weather analysis",
    "fields": [
      "sbcape", "mlcape", "mucape", 
      "stp_fixed", "scp_effective", 
      "srh_01km", "wind_shear_06km"
    ],
    "create_gifs": true
  }
}
```

Run with: `python smart_hrrr_processor.py --latest --workflow severe_weather_suite`

### Custom Filters

Save field combinations in `custom_filters.json`:

```json
{
  "Storm Chaser Core": {
    "description": "Essential parameters for storm chasing",
    "fields": [
      "reflectivity_comp", "sbcape", "effective_shear", 
      "stp_effective", "right_mover_storm_motion"
    ]
  }
}
```

Apply with: `python smart_hrrr_processor.py --latest --filter "Storm Chaser Core"`

## Best Practices

### Field Naming
- Use descriptive, unambiguous names
- Follow meteorological conventions
- Include layer information (e.g., `srh_01km`, `cape_ml`)

### Level Selection
- Choose levels that highlight meteorologically significant values
- Use standard thresholds (e.g., CAPE: 1000, 2500, 4000 J/kg)
- Consider operational forecasting needs

### Colormap Choice
- Match parameter type (diverging for anomalies, sequential for quantities)
- Use established meteorological color conventions
- Ensure accessibility for colorblind users

### Documentation
- Include descriptive titles and units
- Document calculation methodology for derived parameters
- Add comments explaining complex configurations

## Troubleshooting

### Common Issues

**"Missing access configuration"**: Add required access pattern and var/param_id
**"Field not found in GRIB"**: Check variable name and GRIB file contents
**"Invalid colormap"**: Ensure colormap exists in system or custom definitions
**"No data loaded"**: Verify GRIB access pattern matches data structure

### Debugging Tools

**Check field registry**: `python -c "from field_registry import FieldRegistry; r=FieldRegistry(); print(r.list_available_fields())"`

**Inspect GRIB**: `grib_ls -p paramId,shortName,level input.grib2`

**Test single field**: `python smart_hrrr_processor.py --latest --fields field_name --debug`

## Command Reference

### Basic Usage
```bash
# Process latest model run
python smart_hrrr_processor.py --latest

# Specific fields
python smart_hrrr_processor.py --latest --fields sbcape,mlcape,stp_fixed

# By category
python smart_hrrr_processor.py --latest --categories severe,instability

# Custom workflow  
python smart_hrrr_processor.py --latest --workflow storm_chaser_pro

# With GIF creation
python smart_hrrr_processor.py --latest --categories severe --gifs
```

### Advanced Options
```bash
# Specific model run
python smart_hrrr_processor.py 20241201 12 --fields reflectivity_comp

# Limited forecast hours
python smart_hrrr_processor.py --latest --max-hours 6

# Force reprocessing
python smart_hrrr_processor.py --latest --force

# Performance profiling
python smart_hrrr_processor.py --latest --profile --workers 4
```

This system's configuration-driven architecture makes it straightforward to add new meteorological parameters, create custom visualizations, and extend capabilities. The template system and robust data loading provide a solid foundation for operational weather visualization.