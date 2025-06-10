# HRRR Weather Processing - Installation & Setup Guide

Complete installation guide for the HRRR weather processing system on macOS, Linux, and Windows (WSL).

## 🚀 Quick Setup Overview

1. **Install system dependencies** (Python 3.9+, GDAL, ECCODES)
2. **Clone repository** and switch to working branch  
3. **Install Python packages** (scientific stack + geospatial libraries)
4. **Configure environment variables**
5. **Test installation** with sample processing

---

## 📋 Prerequisites

### Python Version
- **Required:** Python 3.9 or higher
- **Recommended:** Python 3.11 for best performance

### System Requirements
- **Memory:** 4GB RAM minimum, 8GB+ recommended for large processing jobs
- **Storage:** 2GB+ for dependencies, additional space for weather data output
- **CPU:** Multi-core recommended for parallel processing

---

## 🍎 macOS Installation

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install System Dependencies
```bash
# Essential tools
brew install git python@3.11

# Geospatial libraries for GRIB processing
brew install gdal proj geos eccodes

# Scientific computing support
brew install hdf5 netcdf
```

### 3. Configure Environment Variables
```bash
# Add to your shell profile (~/.zshrc for zsh or ~/.bash_profile for bash)
echo 'export ECCODES_DIR=/opt/homebrew/lib' >> ~/.zshrc
echo 'export ECCODES_DEFINITION_PATH=/opt/homebrew/share/eccodes/definitions' >> ~/.zshrc
echo 'export PROJ_DATA=/opt/homebrew/share/proj' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc
```

### 4. Apple Silicon (M1/M2) Considerations
```bash
# Ensure you're using native ARM Python
which python3
# Should show: /opt/homebrew/bin/python3

# If you encounter issues, use conda for better M1/M2 support
conda install -c conda-forge cartopy cfgrib xarray matplotlib
```

---

## 🐧 Linux Installation (Ubuntu/Debian)

### 1. Install System Dependencies
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv git

# Install geospatial libraries
sudo apt install libgdal-dev gdal-bin libproj-dev libgeos-dev

# Install ECCODES for GRIB processing
sudo apt install libeccodes-dev libeccodes-tools

# Install HDF5 and NetCDF
sudo apt install libhdf5-dev libnetcdf-dev
```

### 2. Configure Environment (if needed)
```bash
# Usually not needed on Linux, but if ECCODES isn't found:
export ECCODES_DIR=/usr
export ECCODES_DEFINITION_PATH=/usr/share/eccodes/definitions
```

---

## 🪟 Windows Installation (WSL Recommended)

### Option A: Windows Subsystem for Linux (Recommended)
1. **Enable WSL2** and install Ubuntu from Microsoft Store
2. **Follow Linux installation steps** above within WSL
3. **Access files** via `/mnt/c/` for Windows drive integration

### Option B: Native Windows (Advanced)
1. **Install Anaconda/Miniconda** for Python package management
2. **Use conda-forge** for geospatial dependencies:
   ```bash
   conda install -c conda-forge gdal eccodes cartopy cfgrib
   ```

---

## 📦 Repository Setup

### 1. Clone Repository
```bash
# Navigate to your preferred directory
cd ~/Documents  # or wherever you prefer

# Clone the repository
git clone https://github.com/YourUsername/hrrr_com.git
cd hrrr_com
```

### 2. Switch to Working Branch
```bash
# Switch to the active development branch
git checkout smart-processor-v2

# Verify you're on the right branch
git branch
```

### 3. Navigate to Working Directory
```bash
cd complete_src
```

---

## 🐍 Python Environment Setup

### Option A: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv hrrr_env

# Activate environment
source hrrr_env/bin/activate  # Linux/Mac
# OR
hrrr_env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### Option B: Conda Environment
```bash
# Create conda environment
conda create -n hrrr_env python=3.11
conda activate hrrr_env
```

---

## 📚 Python Dependencies Installation

### 1. Core Scientific Libraries
```bash
# Install numerical computing foundation
pip install numpy>=1.20.0

# Install scientific stack
pip install matplotlib>=3.5.0 xarray>=0.20.0 netcdf4>=1.5.0
```

### 2. Geospatial Libraries
```bash
# Install map projections and cartography
pip install cartopy>=0.20.0

# If Cartopy installation fails, try conda:
# conda install -c conda-forge cartopy
```

### 3. GRIB Processing Libraries
```bash
# Install GRIB2 file processing
pip install cfgrib>=0.9.10

# Install ECCODES Python bindings
pip install eccodes>=1.4.0
```

### 4. Additional Utilities
```bash
# System monitoring and HTTP requests
pip install psutil>=5.8.0 requests>=2.25.0

# Units handling for meteorological calculations
pip install pint>=0.19.0

# MetPy for advanced meteorological calculations (optional but recommended)
pip install metpy>=1.6.0
```

### 5. One-Command Installation
```bash
# Install all dependencies at once
pip install numpy matplotlib xarray netcdf4 cartopy cfgrib eccodes psutil requests pint metpy
```

---

## ✅ Installation Verification

### 1. Test Core Dependencies
```bash
python3 -c "
import numpy
import matplotlib
import xarray
import cartopy
import cfgrib
import psutil
print('✅ All core dependencies installed successfully!')
"
```

### 2. Test GRIB Processing
```bash
python3 -c "
import cfgrib
import eccodes
print('✅ GRIB processing libraries ready!')
"
```

### 3. Test System Integration
```bash
# Quick functionality test
python smart_hrrr_processor.py --latest --categories surface --max-hours 1 --debug
```

---

## 🔧 Configuration & Testing

### 1. Test with Sample Data
```bash
# Test with surface weather parameters (fast)
python smart_hrrr_processor.py --latest --categories surface --max-hours 3

# Test with severe weather parameters
python smart_hrrr_processor.py --latest --categories severe --max-hours 1
```

### 2. Check Available Categories
```bash
python3 -c "
from field_registry import FieldRegistry
registry = FieldRegistry()
categories = registry.get_available_categories()
print('Available categories:', categories)
"
```

### 3. Performance Test
```bash
# Test parallel processing
python smart_hrrr_processor.py --latest --categories surface --workers 4 --profile
```

---

## 📁 Directory Structure After Setup

```
~/Documents/hrrr_com/
├── complete_src/                    # Main working directory (cd here)
│   ├── smart_hrrr_processor.py     # Main processing script
│   ├── field_registry.py           # Parameter management
│   ├── parameters/                 # Weather parameter configurations
│   │   ├── surface.json
│   │   ├── severe.json
│   │   ├── smoke.json
│   │   └── personality.json
│   ├── outputs/                    # Generated maps and data
│   │   └── hrrr/
│   │       └── 20250602/
│   │           └── 17z/
│   │               ├── F00/        # Forecast hour directories
│   │               └── logs/       # Processing logs
│   └── README.md                   # This guide
```

---

## 🚨 Troubleshooting

### ECCODES Issues
```bash
# If you get ECCODES-related errors:
brew uninstall eccodes
brew install eccodes
pip uninstall eccodes cfgrib
pip install eccodes cfgrib

# Verify environment variables are set correctly
echo $ECCODES_DIR
echo $ECCODES_DEFINITION_PATH
```

### Cartopy Installation Issues
```bash
# Use conda if pip fails:
conda install -c conda-forge cartopy

# Or try with specific build tools:
pip install --no-binary cartopy cartopy
```

### GDAL/Projection Issues
```bash
# Set PROJ_DATA environment variable
export PROJ_DATA=/usr/share/proj  # Linux
export PROJ_DATA=/opt/homebrew/share/proj  # Mac
```

### Memory Issues
```bash
# Use fewer workers for limited memory systems
python smart_hrrr_processor.py --latest --categories surface --workers 1

# Monitor memory usage during processing
python smart_hrrr_processor.py --latest --categories surface --debug --profile
```

### Permission Issues
```bash
# Ensure write permissions for output directory
chmod -R 755 outputs/

# Check disk space
df -h .
```

---

## ⚡ Performance Optimization

### 1. Worker Configuration
```bash
# Find optimal worker count for your system
for workers in 1 2 4 6; do
    echo "Testing $workers workers..."
    time python smart_hrrr_processor.py --latest --categories surface --max-hours 1 --workers $workers
done
```

### 2. Memory Management
```bash
# Monitor memory usage
python smart_hrrr_processor.py --latest --categories surface --profile --workers 2
```

### 3. Platform-Specific Optimizations

**macOS:**
- Use 2-4 workers typically optimal
- Apple Silicon: prefer conda for scientific libraries
- Intel Macs: standard pip installation works fine

**Linux:**
- Can handle more workers efficiently
- 4-8 workers often optimal on multi-core systems
- Uses system package manager libraries

**Windows/WSL:**
- WSL2 recommended over WSL1 for performance
- 2-4 workers typically optimal
- File I/O may be slower across Windows/Linux boundary

---

## 🎯 Next Steps

1. **Read [USER_GUIDE.md](USER_GUIDE.md)** for complete usage examples
2. **Test different categories**: Start with `--categories surface` then try `severe`, `smoke`, etc.
3. **Set up monitoring**: Use `--latest` mode for real-time processing  
4. **Explore personality composites**: Try `--categories personality` for creative weather indices
5. **Performance tune**: Use `--profile` to optimize for your system

## 🆘 Getting Help

- **Check logs**: Processing logs are saved in `outputs/hrrr/*/logs/` for debugging
- **Use debug mode**: Add `--debug` flag for verbose output
- **Monitor resources**: Use `--profile` to understand performance bottlenecks
- **Start small**: Test with `--max-hours 1` and single categories first

Your system should now be ready to generate professional-quality weather maps from live HRRR data! 🌤️