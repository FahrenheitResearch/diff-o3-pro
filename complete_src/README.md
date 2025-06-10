# HRRR Complete Weather Processing System

A modern, extensible framework for processing High-Resolution Rapid Refresh (HRRR) weather model data into professional-quality meteorological maps and visualizations.

## 🚀 Quick Start

```bash
cd complete_src

# Process latest weather data with severe weather parameters
python smart_hrrr_processor.py --latest --categories severe

# Process specific model run with all surface weather
python smart_hrrr_processor.py 20250602 17 --categories surface,reflectivity

# Monitor live data with real-time processing
python smart_hrrr_processor.py --latest --check-interval 30
# Run a workflow preset with GIF creation
python smart_hrrr_processor.py --latest --workflow all_maps_gifs
# Toggle GIF creation manually
python smart_hrrr_processor.py --latest --categories severe --max-hours 6 --gifs

# Process a specific hour range with a preset
python smart_hrrr_processor.py 20250602 12 --hours 6-12 --workflow top1
```

## 📊 What This System Does

### Core Capabilities
- **50+ meteorological parameters** across 9 categories (surface, severe weather, smoke, etc.)
- **Real-time processing** of live HRRR model data
- **Professional visualizations** with SPC-style quality and colormaps
- **Smart duplicate detection** - only processes missing products
- **Parallel processing** for optimal performance
- **Live monitoring** with automatic new data detection

### Output Quality
- Clean, professional SPC-style maps
- Native HRRR resolution (1059×1799 pixels)
- Proper Lambert Conformal projection
- Publication-ready formatting
- Custom meteorological color scales

### Performance Metrics
- **96.6% success rate** on HRRR parameters
- **Smart processing** - skips existing files automatically
- **Parallel workers** for multi-core optimization
- **Memory efficient** streaming processing

## 🏗️ System Architecture

### Modern Refactored Design
This system has evolved from a monolithic 545-line script into a modern, modular framework:

| Component | Purpose |
|-----------|---------|
| **`smart_hrrr_processor.py`** | Main processing engine with live monitoring |
| **`field_registry.py`** | Central parameter management system |
| **`field_templates.py`** | Template inheritance system for easy parameter addition |
| **`parameters/`** | JSON configuration files organized by category |
| **`derived_parameters.py`** | Advanced calculations and personality composites |

### Key Improvements Over Original
- **10x easier** parameter addition (JSON config vs. code editing)
- **Modular components** replace 545-line monolith
- **Category organization** with logical grouping
- **Extensible design** for future enhancements
- **Robust error handling** with graceful failures

## 📁 Output Organization

```
outputs/
└── hrrr/
    └── 20250602/          # Date (YYYYMMDD)
        └── 17z/           # Model run hour
            ├── F00/       # Forecast hour directories
            │   ├── sbcape_f00_REFACTORED.png
            │   ├── reflectivity_comp_f00_REFACTORED.png
            │   └── total_column_smoke_f00_REFACTORED.png
            ├── F01/
            ├── F02/
            └── logs/      # Processing logs with detailed status
```

## 🌤️ Available Weather Parameters

### Categories
- **atmospheric**: Temperature, humidity, pressure, boundary layer
- **surface**: 2m/10m measurements and derived parameters  
- **instability**: CAPE, CIN, lifted index, convective parameters
- **severe**: Tornado parameters, supercell composite, wind shear
- **precipitation**: Precipitation rate, accumulation, snow
- **reflectivity**: Radar reflectivity at different levels
- **smoke**: Smoke concentration, visibility, fire weather
- **derived**: Advanced calculations and composites
- **personality**: Custom weather indices with unique perspectives

### Example Parameters
- **Surface-Based CAPE/CIN** - Convective instability measures
- **Storm Relative Helicity** - Tornado development potential
- **Composite Reflectivity** - Radar precipitation intensity
- **Near-Surface Smoke** - Air quality and visibility impact
- **Supercell Composite Parameter** - Severe thunderstorm potential
- **Personality Composites** - Creative indices like "Kazoo MAXX" and "Destroyer Reality Check"

### Saved Filters
Define custom filters in `custom_filters.json`. A filter lists specific categories and field names to process. Use `--filter` to apply one from the command line or `--fields` to specify products directly.

```bash
python smart_hrrr_processor.py --latest --filter "Severe Weather Core"
python smart_hrrr_processor.py --latest --fields sbcape,reflectivity_comp
```

### Workflow Presets
Define reusable processing presets in `custom_workflows.json` and run them with `--workflow`.

```bash
# Generate all maps and GIFs
python smart_hrrr_processor.py --latest --workflow all_maps_gifs

# Top parameter from each category
python smart_hrrr_processor.py 20250602 12 --workflow top1
# Storm chaser focused maps (~25 severe weather fields)
python smart_hrrr_processor.py --latest --workflow storm_chaser_pro
# Quick radar, precip and surface conditions for casual viewers
python smart_hrrr_processor.py --latest --workflow casual_viewer
# Fun personality composites for Discord weather nerds
python smart_hrrr_processor.py --latest --workflow discord_weather_nerd
```

## 🔧 Usage Modes

### 1. Live Monitoring
```bash
# Monitor latest model runs automatically
python smart_hrrr_processor.py --latest --categories severe,smoke
```

### 2. Specific Model Runs
```bash
# Process specific date/hour with custom parameters
python smart_hrrr_processor.py 20250602 17 --max-hours 12 --workers 4

# Process a custom forecast-hour range
python smart_hrrr_processor.py 20250602 17 --hours 6-12
```

### 3. Category-Focused Processing
```bash
# Process only storm-chasing parameters for speed
python smart_hrrr_processor.py --latest --categories severe,reflectivity
```

### 4. Performance Optimization
```bash
# Enable profiling for performance analysis
python smart_hrrr_processor.py --latest --profile --workers 4
```

## 🎯 Common Use Cases

### Storm Chasing
```bash
python smart_hrrr_processor.py --latest --categories severe,reflectivity --max-hours 6
```

### Fire Weather Monitoring
```bash
python smart_hrrr_processor.py --latest --categories smoke --check-interval 15
```

### Aviation Weather
```bash
python smart_hrrr_processor.py --latest --categories surface,atmospheric,smoke
```

### Research Analysis
```bash
python smart_hrrr_processor.py 20250602 12 --max-hours 48 --workers 6
```

## 🔍 Smart Features

### Duplicate Detection
- Automatically skips existing PNG files
- Dramatically speeds up re-runs
- Use `--force` to regenerate everything

### Auto-Detection
- **6-hourly runs** (00Z, 06Z, 12Z, 18Z): Process F00-F48
- **Hourly runs** (all others): Process F00-F18
- Override with `--max-hours` or `--hours` parameter

### Live Monitoring
- Checks for new data every 30 seconds (configurable)
- Processes forecast hours as they become available
- Automatic cycle detection and switching

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Installation and configuration guide
- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete usage documentation with examples
- **[TECHNICAL.md](TECHNICAL.md)** - System architecture and implementation details
- **[PERSONALITY.md](PERSONALITY.md)** - Personality weather composites guide
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Future enhancements and contribution guide
- **[GIT_RECOVERY.md](GIT_RECOVERY.md)** - Project-specific git procedures

## ⚡ Performance

- **Parallel processing** with configurable worker count
- **Memory efficient** for long-running monitoring
- **Smart caching** and duplicate detection
- **Profiling tools** for optimization analysis
- **Error recovery** continues processing despite individual failures

## 🌟 What Makes This Special

1. **Professional Quality**: SPC-style maps with proper projections and color scales
2. **Real-Time Ready**: Live monitoring with automatic new data detection
3. **Extensible Design**: Add new parameters via JSON configuration
4. **Performance Optimized**: Parallel processing with smart duplicate detection
5. **User Focused**: Clear documentation and intuitive command structure
6. **Creative Extensions**: Personality composites add fun to serious meteorology

This system transforms raw HRRR GRIB2 files into production-ready meteorological visualizations with minimal effort and maximum flexibility.

---

**Ready to generate professional weather maps from live HRRR data!** 🌤️⚡

See **[USER_GUIDE.md](USER_GUIDE.md)** for detailed usage examples and **[SETUP.md](SETUP.md)** to get started.