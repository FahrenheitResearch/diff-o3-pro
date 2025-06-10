# HRRR Processing - Complete User Guide

Comprehensive guide covering all ways to use the smart HRRR processing system for generating weather maps and visualizations.

## 🚀 Quick Reference

```bash
# Process latest model run with auto-detection
python smart_hrrr_processor.py --latest

# Process specific model run  
python smart_hrrr_processor.py 20250602 17

# Process with specific categories only
python smart_hrrr_processor.py 20250602 17 --categories severe,smoke

# Live monitoring with real-time processing
python smart_hrrr_processor.py --latest --check-interval 30

# Performance profiling enabled
python smart_hrrr_processor.py --latest --profile --workers 4

# Run a workflow preset with GIFs
python smart_hrrr_processor.py --latest --workflow all_maps_gifs

# Custom hour range
python smart_hrrr_processor.py 20250602 12 --hours 6-12 --workflow top1
# Manually enable GIF creation
python smart_hrrr_processor.py --latest --categories severe --max-hours 6 --gifs
```

### Filters
Save reusable filters in `custom_filters.json`. Apply a filter from the command line with `--filter` or provide specific fields with `--fields`.

```bash
python smart_hrrr_processor.py --latest --filter "Severe Weather Core"
python smart_hrrr_processor.py --fields sbcape,reflectivity_comp --latest
```

### Workflow Presets
Presets in `custom_workflows.json` bundle field selections and optional GIF creation. Use `--workflow` to run them quickly.

```bash
# All maps with GIFs
python smart_hrrr_processor.py --latest --workflow all_maps_gifs

# Important parameters only
python smart_hrrr_processor.py 20250602 12 --workflow important
# Storm chaser focused maps (~25 severe weather fields)
python smart_hrrr_processor.py --latest --workflow storm_chaser_pro
# Quick radar, precip and surface conditions
python smart_hrrr_processor.py --latest --workflow casual_viewer
# Discord weather nerd specials
python smart_hrrr_processor.py --latest --workflow discord_weather_nerd
```

---

## 📁 Understanding Output Structure

All outputs are organized in a clean, predictable directory structure:

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

Each forecast hour directory contains PNG files with the naming pattern:
`{parameter_name}_f{XX}_REFACTORED.png`

---

## 🎯 Basic Usage

### 1. Process Complete Model Run
```bash
# Process 17Z run - automatically detects F00-F18 for hourly runs
python smart_hrrr_processor.py 20250602 17

# Process 12Z run - automatically detects F00-F48 for 6-hourly runs  
python smart_hrrr_processor.py 20250602 12
```

**Auto-detection rules:**
- **6-hourly runs** (00Z, 06Z, 12Z, 18Z): Process F00-F48
- **Hourly runs** (all other hours): Process F00-F18
- Override with `--max-hours` or `--hours`

### 2. Specify Forecast Hour Range
```bash
# Process only first 12 forecast hours
python smart_hrrr_processor.py 20250602 17 --max-hours 12

# Process only first 6 forecast hours  
python smart_hrrr_processor.py 20250602 17 --max-hours 6

# Process up to 48 hours for any run
python smart_hrrr_processor.py 20250602 17 --max-hours 48

# Custom hour span
python smart_hrrr_processor.py 20250602 17 --hours 6-12
```

### 3. Live Monitoring Mode
```bash
# Monitor latest model run automatically
python smart_hrrr_processor.py --latest

# Check for new data every 60 seconds
python smart_hrrr_processor.py --latest --check-interval 60

# Monitor with specific categories only
python smart_hrrr_processor.py --latest --categories severe,smoke
```

**How live monitoring works:**
- Automatically detects the most recent HRRR cycle
- Checks every 30 seconds (default) for new forecast hours
- Processes them immediately as they become available
- Stops when all expected forecast hours are processed

---

## 🌤️ Category-Based Processing

### Available Categories
- **atmospheric**: Basic atmospheric parameters (temperature, humidity, pressure)
- **surface**: Surface-level measurements and derived parameters  
- **instability**: CAPE, CIN, lifted index, convective parameters
- **severe**: Tornado parameters, supercell composite, wind shear
- **precipitation**: Precipitation rate, accumulation, snow
- **reflectivity**: Radar reflectivity at different levels
- **smoke**: Smoke concentration, visibility, fire weather
- **derived**: Custom derived parameters and special calculations
- **personality**: Creative weather indices with unique perspectives

### Category Examples
```bash
# Process only severe weather parameters (fastest for storm chasing)
python smart_hrrr_processor.py 20250602 17 --categories severe

# Process only smoke-related products
python smart_hrrr_processor.py 20250602 17 --categories smoke

# Process multiple categories
python smart_hrrr_processor.py 20250602 17 --categories surface,reflectivity,severe

# Process everything (all categories)
python smart_hrrr_processor.py 20250602 17
```

### Quick Product Selection Examples
```bash
# Storm chasing essentials
python smart_hrrr_processor.py --latest --categories severe,reflectivity

# Fire weather monitoring  
python smart_hrrr_processor.py --latest --categories smoke

# Aviation weather package
python smart_hrrr_processor.py --latest --categories surface,atmospheric,smoke

# Current conditions check
python smart_hrrr_processor.py --latest --max-hours 6 --categories surface,reflectivity
```

---

## ⚡ Performance & Efficiency

### Parallel Processing
```bash
# Use 4 parallel workers (faster on multi-core systems)
python smart_hrrr_processor.py 20250602 17 --workers 4

# Use 6 workers for maximum speed
python smart_hrrr_processor.py --latest --workers 6

# Use 1 worker (default, most memory-efficient)
python smart_hrrr_processor.py 20250602 17 --workers 1
```

**Performance notes:**
- More workers = faster processing but higher memory usage
- CPU-bound processing - diminishing returns beyond 4-6 workers
- System automatically limits to available CPU cores

### Smart Duplicate Detection
```bash
# Skip already processed products (default behavior)
python smart_hrrr_processor.py 20250602 17

# Force reprocess everything (ignores existing files)
python smart_hrrr_processor.py 20250602 17 --force
```

**How it works:**
- Automatically detects existing PNG files in output directories
- Only processes missing products
- Dramatically speeds up re-runs and partial processing
- Use `--force` to regenerate everything from scratch

---

## 📊 Performance Profiling

### Enable Profiling
```bash
# Basic profiling
python smart_hrrr_processor.py --latest --categories severe --profile

# Detailed profiling with custom settings
python smart_hrrr_processor.py 20250602 17 --profile --profile-interval 0.5 --profile-output my_analysis.json
```

### What Gets Profiled
- **Memory usage** (system and process)
- **CPU utilization** (system and process)  
- **I/O operations** (read/write bytes and counts)
- **Function call timings** (individual function performance)
- **Phase timing** (setup, processing, cleanup phases)

### Understanding Profiling Output

**Live Output During Processing:**
```
🔍 Starting performance profiling (interval: 1.0s)
📊 Starting phase: model_run_setup
✅ Phase 'model_run_setup' completed in 0.42s
📊 Starting phase: forecast_hour_processing
✅ Phase 'forecast_hour_processing' completed in 127.84s
⏹️ Profiling stopped. Total runtime: 128.73s
```

**Summary Report:**
```
🔍 PERFORMANCE PROFILING SUMMARY
============================================================
📊 Total Runtime: 128.73 seconds

💾 Memory Usage:
  Peak: 2847.3 MB
  Average: 2234.8 MB
  Growth: 423.1 MB

🖥️ CPU Usage:
  Average: 67.2%
  Peak: 94.8%

⏱️ Phase Timings:
  forecast_hour_processing    127.84s (99.3%)
  model_run_setup               0.42s ( 0.3%)

🚀 Top Functions by Time:
  1. process_forecast_hour_smart: 125.22s (97.3%) [6 calls]
  2. check_existing_products: 8.45s (6.6%) [6 calls]
  3. check_cycle_availability: 2.33s (1.8%) [1 calls]
```

### Optimization Analysis

**Memory Issues:**
```bash
# Profile memory-intensive processing
python smart_hrrr_processor.py 20250602 12 --max-hours 48 --profile --workers 1

# Red flags to watch for:
# - Memory growth >500MB = potential memory leak
# - Peak memory >8GB = consider smaller batches
# - Memory efficiency <60% = large spikes detected
```

**CPU Bottlenecks:**
```bash
# Test different worker counts
python smart_hrrr_processor.py --latest --categories severe --workers 1 --profile
python smart_hrrr_processor.py --latest --categories severe --workers 4 --profile

# Optimization signals:
# - CPU <40% = increase workers
# - CPU >95% = reduce workers or processing load
# - High variance = I/O bound (not CPU bound)
```

**Category-Specific Profiling:**
```bash
# Profile individual categories to find expensive ones
python smart_hrrr_processor.py --latest --categories atmospheric --profile
python smart_hrrr_processor.py --latest --categories instability --profile  
python smart_hrrr_processor.py --latest --categories reflectivity --profile
python smart_hrrr_processor.py --latest --categories severe --profile
python smart_hrrr_processor.py --latest --categories smoke --profile
```

---

## 🌪️ Real-World Workflow Examples

### Storm Chasing Workflow
```bash
# Morning: Check overnight model runs
python smart_hrrr_processor.py 20250602 00 --categories severe,reflectivity
python smart_hrrr_processor.py 20250602 06 --categories severe,reflectivity

# Rapid storm analysis
python smart_hrrr_processor.py --latest --categories severe,reflectivity --max-hours 6

# Live storm tracking (fast updates)
python smart_hrrr_processor.py --latest --categories severe,reflectivity --check-interval 20
```

### Fire Weather Monitoring
```bash
# Quick current smoke conditions
python smart_hrrr_processor.py --latest --categories smoke --max-hours 6

# Extended fire weather monitoring
python smart_hrrr_processor.py --latest --categories smoke --check-interval 15

# Full fire weather assessment
python smart_hrrr_processor.py 20250602 17 --categories smoke,surface,atmospheric --max-hours 24
```

### Aviation Weather Package
```bash
# Current conditions for flight planning
python smart_hrrr_processor.py --latest --categories surface,atmospheric,smoke --max-hours 12

# Detailed route weather
python smart_hrrr_processor.py 20250602 12 --categories surface,atmospheric,precipitation --max-hours 48
```

### Research & Analysis
```bash
# Comprehensive case study analysis
python smart_hrrr_processor.py 20250602 12 --max-hours 48 --workers 6

# High-impact weather analysis
python smart_hrrr_processor.py 20250602 12 --categories severe,precipitation,instability --max-hours 48

# Performance optimization research
python smart_hrrr_processor.py 20250602 12 --max-hours 48 --profile --workers 1
python smart_hrrr_processor.py 20250602 12 --max-hours 48 --profile --workers 4
```

### Automated Monitoring Scripts
```bash
#!/bin/bash
# Cron job for automated severe weather alerts (every 15 minutes)
python smart_hrrr_processor.py --latest --categories severe --check-interval 30

#!/bin/bash  
# Fire weather dashboard update (every 10 minutes)
python smart_hrrr_processor.py --latest --categories smoke --max-hours 12

#!/bin/bash
# Daily research data collection
DATE=$(date -u +%Y%m%d)
for HOUR in 00 06 12 18; do
    python smart_hrrr_processor.py $DATE $HOUR --workers 4
done
```

---

## 🔧 Advanced Usage

### Single Forecast Hour Processing
```bash
# Process just F00 (current conditions)
python smart_hrrr_processor.py 20250602 17 --max-hours 0

# Process F06 and save to the default output directory
python smart_hrrr_processor.py 20250602 17 --max-hours 6

# Process F12 with specific categories
python smart_hrrr_processor.py 20250602 17 --max-hours 12 --categories severe,smoke
```

### Debug and Troubleshooting
```bash
# Enable detailed logging
python smart_hrrr_processor.py 20250602 17 --debug

# Check what forecast hours are available
python smart_hrrr_processor.py --latest --debug

# Clean up old files before processing
python smart_hrrr_processor.py 20250602 17 --cleanup
```

### Custom Processing Configurations
```bash
# Process with specific geographic focus
python smart_hrrr_processor.py --latest --categories severe --max-hours 12 --workers 2

# Memory-conservative processing
python smart_hrrr_processor.py --latest --categories surface --workers 1 --max-hours 6

# High-speed processing for real-time applications
python smart_hrrr_processor.py --latest --categories smoke --workers 4 --check-interval 15
```

---

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

**High Memory Usage:**
```bash
# Diagnose memory problems
python smart_hrrr_processor.py --latest --categories surface --max-hours 1 --profile

# Solutions:
# - Reduce forecast hours: --max-hours 6
# - Use single worker: --workers 1  
# - Process categories separately
```

**Slow Processing:**
```bash
# Identify bottlenecks
python smart_hrrr_processor.py --latest --categories surface --max-hours 1 --profile --debug

# Solutions:
# - Increase workers: --workers 4
# - Use faster categories: --categories surface,severe
# - Check network connection for data downloads
```

**Missing Output Files:**
```bash
# Check processing logs
ls outputs/hrrr/20250602/17z/logs/

# View latest log
cat outputs/hrrr/20250602/17z/logs/processing_*.log

# Force regeneration
python smart_hrrr_processor.py 20250602 17 --force
```

**Data Availability Issues:**
```bash
# Check what's available
python smart_hrrr_processor.py --latest --debug

# Try different time ranges
python smart_hrrr_processor.py --latest --max-hours 3

# Test with older model run
python smart_hrrr_processor.py 20250602 12
```

### Performance Tips

1. **Start with categories** - only generate what you need for faster processing
2. **Use live monitoring** for most efficient real-time applications
3. **Smart duplicate detection** means you can safely re-run commands
4. **Parallel workers** help on multi-core systems, but 4-6 is usually optimal
5. **Profile first** - understand your system's performance characteristics

### Best Practices

**For Storm Chasing:**
- Use `--categories severe,reflectivity` for essential parameters
- Set `--max-hours 6` for near-term focus
- Enable `--profile` to optimize update speed

**For Fire Weather:**
- Use `--categories smoke` for rapid smoke-only updates
- Set `--check-interval 15` for frequent monitoring
- Include `surface` category for wind information

**For Research:**
- Use `--workers 4-6` for parallel processing
- Enable `--profile` for performance analysis
- Process full forecast ranges with `--max-hours 48`

**For Operational Use:**
- Set up automated scripts with appropriate check intervals
- Use category-specific processing for targeted applications
- Monitor logs for error detection and system health

---

## 🎬 Animated GIF Creation

The system includes powerful tools for creating animated GIFs from forecast sequences, perfect for time-lapse weather animations and social media sharing.

### Single Parameter GIF Creation

Create GIFs for individual weather parameters:

```bash
# Create GIF from any parameter file (auto-detects sequence)
python hrrr_gif_maker.py "outputs/hrrr/20250604/18z/F00/personality/destroyer_reality_check_f00_REFACTORED.png"

# Create GIF with custom settings
python hrrr_gif_maker.py "destroyer_reality_check_f00_REFACTORED.png" --date 20250604 --hour 18z --duration 300 --max-hours 24

# Interactive mode to browse and select parameters
python hrrr_gif_maker.py --interactive
```

**Key Features:**
- **Auto-detection** - Automatically finds all forecast hours for the parameter
- **Smart parsing** - Extracts date, hour, and category from file paths
- **Flexible input** - Works with full paths or just filenames
- **Custom timing** - Adjustable frame duration and forecast hour limits

### Batch GIF Creation

Create GIFs for all parameters in a model run:

```bash
# Create GIFs for all categories (can generate 50+ animations)
python create_gifs.py 20250604 18z

# Create GIFs for specific categories only
python create_gifs.py 20250604 18z --categories personality,severe

# Create shorter, faster animations
python create_gifs.py 20250604 18z --max-hours 12 --duration 300
```

**Batch Processing Features:**
- **Auto-discovery** - Finds all processed parameters automatically
- **Organized output** - Creates animations/ directory with category subfolders
- **Smart skipping** - Skips existing GIFs to avoid reprocessing
- **Progress tracking** - Shows creation progress and file sizes

### GIF Customization Options

**Frame Duration:**
- `--duration 500` (default) - 0.5 seconds per frame
- `--duration 300` - Faster animations (0.3s per frame)
- `--duration 1000` - Slower animations (1.0s per frame)

**Forecast Hours:**
- `--max-hours 48` (default) - Full forecast range
- `--max-hours 12` - Short-term focus (12 hours)
- `--max-hours 6` - Immediate forecast only

**Output Control:**
- `--output custom_name.gif` - Specify custom filename
- Auto-generated names include parameter, date, and timestamp

### Animation Examples by Category

**Personality Composites:**
```bash
# Destroyer Reality Check evolution
python hrrr_gif_maker.py "outputs/hrrr/20250604/18z/F00/personality/destroyer_reality_check_f00_REFACTORED.png" --max-hours 18

# All personality composites for model run
python create_gifs.py 20250604 18z --categories personality --duration 400
```

**Severe Weather:**
```bash
# Storm evolution analysis
python create_gifs.py 20250604 17z --categories severe,reflectivity --max-hours 12

# Tornado parameter tracking
python hrrr_gif_maker.py "srh_01km_f00_REFACTORED.png" --date 20250604 --hour 17z --duration 350
```

**Advanced/Derived Parameters:**
```bash
# Heat stress analysis (wet bulb temperature)
python hrrr_gif_maker.py "outputs/hrrr/20250604/20z/F00/advanced/wet_bulb_temp_f00_REFACTORED.png" --max-hours 24

# All advanced parameters for model run
python create_gifs.py 20250604 20z --categories advanced --duration 400
```

**Smoke Tracking:**
```bash
# Fire/smoke evolution
python create_gifs.py 20250604 15z --categories smoke --max-hours 24 --duration 600

# Near-surface smoke concentration
python hrrr_gif_maker.py "near_surface_smoke_f00_REFACTORED.png" --date 20250604 --hour 15z
```

### Interactive Mode Walkthrough

The interactive mode provides a guided experience:

```bash
python hrrr_gif_maker.py --interactive
```

**Interactive Steps:**
1. **Enter date** (YYYYMMDD format)
2. **Enter model hour** (e.g., 18z)
3. **Select category** from discovered categories
4. **Select parameter** from available parameters
5. **Configure settings** (duration, max hours)
6. **Generate GIF** with progress feedback

### Output Organization

GIFs are organized in a clean structure:
```
outputs/hrrr/20250604/18z/
├── animations/                    # Created by batch processor
│   ├── personality/
│   │   ├── destroyer_reality_check_20250604_18z_animation.gif
│   │   ├── kazoo_maxx_20250604_18z_animation.gif
│   │   └── seqouigrove_weird_west_20250604_18z_animation.gif
│   ├── severe/
│   │   ├── srh_01km_20250604_18z_animation.gif
│   │   └── wind_shear_06km_20250604_18z_animation.gif
│   └── reflectivity/
└── individual_parameter_name_timestamp.gif  # Created by single tool
```

### Performance Tips

**File Size Optimization:**
- Shorter sequences (`--max-hours 12`) create smaller files
- Longer frame duration (`--duration 600`) allows more compression
- Personality composites typically 0.5-2.0 MB per GIF
- Reflectivity animations can be larger (2-5 MB)

**Processing Speed:**
- Use `--categories` to process only needed categories
- Batch processing is more efficient than individual parameter creation
- GIF creation adds ~5-10 seconds per parameter

**Storage Management:**
- Full 48-hour animations can use significant disk space
- Consider creating separate directories for different animation types
- Use `--max-hours 6` for quick preview animations

### Workflow Integration

**Storm Chasing Workflow:**
```bash
# Process model run
python smart_hrrr_processor.py 20250602 17z --categories severe,reflectivity

# Create quick 6-hour animations for immediate use
python create_gifs.py 20250602 17z --categories severe,reflectivity --max-hours 6 --duration 300

# Create full animations for analysis
python create_gifs.py 20250602 17z --categories severe,reflectivity --max-hours 18
```

**Social Media Sharing:**
```bash
# Create fast, engaging animations
python create_gifs.py 20250602 18z --categories personality --duration 250 --max-hours 12

# Create dramatic storm evolution GIFs
python hrrr_gif_maker.py "reflectivity_comp_f00_REFACTORED.png" --date 20250602 --hour 17z --duration 400 --max-hours 18
```

**Research Analysis:**
```bash
# Create comprehensive animation sets
python create_gifs.py 20250602 12z --max-hours 48 --duration 500

# Focus on specific parameters
python hrrr_gif_maker.py "sbcape_f00_REFACTORED.png" --date 20250602 --hour 12z --max-hours 48
```

The GIF creation tools turn static weather maps into engaging animations that reveal the evolution of meteorological phenomena over time! 🎬⚡

---

This comprehensive guide covers all aspects of using the HRRR processing system efficiently for your specific meteorological applications! 🌤️⚡