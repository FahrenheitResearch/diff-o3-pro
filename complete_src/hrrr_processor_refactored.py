#!/usr/bin/env python3
"""
HRRR Processor - Refactored Extensible Version
Generates HRRR maps using configuration-driven field definitions
Easily extensible for new parameters and variables
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
import xarray as xr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from field_registry import FieldRegistry
from derived_params import DerivedParameters, compute_derived_parameter

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class HRRRProcessor:
    """HRRR data processor with extensible field configurations"""
    
    def __init__(self, config_dir: Path = None):
        """Initialize HRRR processor
        
        Args:
            config_dir: Directory containing parameter configuration files
        """
        self.registry = FieldRegistry(config_dir)
        self.colormaps = self.create_spc_colormaps()
        
    def create_spc_colormaps(self):
        """Create SPC-style colormaps"""
        colormaps = {}
        
        # NWS Reflectivity
        ref_colors = ['#646464', '#04e9e7', '#019ff4', '#0300f4', '#02fd02',
                      '#01c501', '#008e00', '#fdf802', '#e5bc00', '#fd9500',
                      '#fd0000', '#d40000', '#bc0000', '#f800fd', '#9854c6']
        colormaps['NWSReflectivity'] = LinearSegmentedColormap.from_list('NWSRef', ref_colors)
        
        # CAPE - Yellow to Red
        cape_colors = ['#ffff00', '#ffd700', '#ff8c00', '#ff4500', '#ff0000', '#dc143c', '#8b0000']
        colormaps['CAPE'] = LinearSegmentedColormap.from_list('CAPE', cape_colors)
        
        # CIN - Blues
        cin_colors = ['#e6f3ff', '#cce7ff', '#99d6ff', '#66c2ff', '#3399ff', '#0080ff', '#0066cc', '#004d99']
        colormaps['CIN'] = LinearSegmentedColormap.from_list('CIN', cin_colors)
        
        # Lifted Index - Blue to Red
        li_colors = ['#0000ff', '#4169e1', '#87ceeb', '#f0f8ff', '#ffffff', '#ffe4e1', '#ffa07a', '#ff4500', '#ff0000']
        colormaps['LiftedIndex'] = LinearSegmentedColormap.from_list('LiftedIndex', li_colors)
        
        # Hail - Green to Purple
        hail_colors = ['#00ff00', '#32cd32', '#9acd32', '#ffff00', '#ffa500', '#ff4500', '#ff0000', '#800080']
        colormaps['Hail'] = LinearSegmentedColormap.from_list('Hail', hail_colors)
        
        # NOAA Smoke - Light Blue to Blue to Green to Yellow to Orange to Red to Purple (less transparent)
        smoke_colors = ['#e6f3ff', '#87ceeb', '#90ee90', '#ffff00', '#ffa500', '#ff4500', '#ff0000', '#800080']
        colormaps['NOAASmoke'] = LinearSegmentedColormap.from_list('NOAASmoke', smoke_colors)
        
        # Tornado Diagnostics Colormaps
        
        # STP colormap - light below STP=1, then increasingly warm colors for higher values
        stp_colors = ['#f7f7f7', '#e0e0e0', '#cccccc', '#ffeda0', '#fed976', '#feb24c', 
                     '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
        colormaps['STP'] = LinearSegmentedColormap.from_list('STP', stp_colors)
        
        # SCP colormap - diverging centered on zero (blue for negative/left-moving, red for positive/right-moving)
        scp_colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                     '#fdbf6f', '#fd8d3c', '#e31a1c', '#b10026', '#67001f']
        colormaps['SCP'] = LinearSegmentedColormap.from_list('SCP', scp_colors)
        
        # EHI colormap - diverging centered on zero (blue for negative/left-moving, red for positive/right-moving)
        ehi_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                     '#fdbf6f', '#fd8d3c', '#e31a1c', '#b10026', '#67001f']
        colormaps['EHI'] = LinearSegmentedColormap.from_list('EHI', ehi_colors)
        
        # Storm-Relative Helicity - red scales for rotation
        srh_colors = ['#ffeeee', '#ffcccc', '#ffaaaa', '#ff8888', '#ff6666', 
                     '#ff4444', '#ff2222', '#ff0000', '#dd0000', '#bb0000', '#990000']
        colormaps['SRH'] = LinearSegmentedColormap.from_list('SRH', srh_colors)
        
        # Wind Shear - viridis-like for wind magnitude
        shear_colors = ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', 
                       '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825']
        colormaps['WindShear'] = LinearSegmentedColormap.from_list('WindShear', shear_colors)
        
        # BRN colormap - emphasizes the 10-45 "supercell sweet spot"
        # Light blue for extreme shear (<10), green-yellow for supercells (10-45), orange-red for weak shear (>50)
        brn_colors = ['#08306b', '#4292c6', '#9ecae1', '#41ab5d', '#78c679', '#addd8e', 
                     '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
        colormaps['BRN'] = LinearSegmentedColormap.from_list('BRN', brn_colors)
        
        # LCL colormap - inverted (low LCL = good = green/blue, high LCL = bad = yellow/red)
        lcl_colors = ['#004529', '#238b45', '#41ab5d', '#74c476', '#a1d99b', 
                     '#c7e9c0', '#edf8e9', '#fff7ec', '#fee8c8', '#fdd49e', 
                     '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000']
        colormaps['LCL'] = LinearSegmentedColormap.from_list('LCL', lcl_colors)
        
        # Personality Composite Colormaps
        
        # Seqouigrove Weird-West Composite - Desert vibes with moisture pop
        # Blue (boring dry) -> White (neutral) -> Yellow/Orange (getting interesting) -> Red/Purple (peak weirdness)
        sw2c_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                      '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026', '#67001f']
        colormaps['SeqouigroveWeirdWest'] = LinearSegmentedColormap.from_list('SW2C', sw2c_colors)
        
        # K-MAX Composite - Pure hype: everything-to-eleven
        # Green (meh) -> Yellow (getting spicy) -> Orange (serious business) -> Red (HYPE) -> Purple (legendary)
        kmax_colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', 
                      '#fdae61', '#f46d43', '#d73027', '#a50026', '#762a83']
        colormaps['KazooMAXX'] = LinearSegmentedColormap.from_list('KMAX', kmax_colors)
        
        # Seqouiagrove Thermal Range - Diurnal temperature spread vibes
        # Cool blue (small range) -> Warm amber (moderate) -> Hot orange/red (big swings) -> Deep crimson (epic ranges)
        thermal_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fdbf6f', 
                         '#fd8d3c', '#e31a1c', '#bd0026', '#800026', '#67001f']
        colormaps['SeqouiagroveThermal'] = LinearSegmentedColormap.from_list('SThermal', thermal_colors)
        
        # Destroyer Reality-Check - Anti-hype truth colormap
        # Dark gray (hype/bust) -> Red (weak) -> Orange (marginal) -> Yellow (decent) -> Green (legit) -> Bright green (chase-worthy)
        destroyer_colors = ['#2b2b2b', '#636363', '#969696', '#cc4c02', '#fd8d3c', 
                           '#fecc5c', '#ffffb2', '#c7e9b4', '#7fcdbb', '#2c7fb8']
        colormaps['DestroyerReality'] = LinearSegmentedColormap.from_list('DReality', destroyer_colors)
        
        # Samuel Outflow Propensity - Cold pool science colormap  
        # Light blue (weak outflow) -> White (marginal) -> Yellow (moderate) -> Orange (strong) -> Red (violent outflow) -> Dark red (gust front city)
        samuel_colors = ['#c6dbef', '#9ecae1', '#6baed6', '#f7f7f7', '#fee391', 
                        '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']
        colormaps['SamuelOutflow'] = LinearSegmentedColormap.from_list('SOutflow', samuel_colors)
        
        # Mason-Flappity Bayou Buzz - Gulf Coast high-impact weather composite
        # Navy (synoptically benign) -> Blue (seabreeze) -> Teal (ordinary storms) -> Yellow (watch-worthy) -> Orange (enhanced severe) -> Red (high-impact events) -> Deep red (hurricane/historic)
        mf_buzz_colors = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                         '#fee391', '#fec44f', '#fe9929', '#d94701', '#a63603']
        colormaps['MasonFlappityBuzz'] = LinearSegmentedColormap.from_list('MFBuzz', mf_buzz_colors)
        
        
        return colormaps

    def download_hrrr_file(self, cycle, forecast_hour, output_dir, file_type='wrfprs'):
        """Download HRRR file with fallback to AWS S3 for historical data
        
        Args:
            cycle: HRRR cycle (YYYYMMDDHH)
            forecast_hour: Forecast hour
            output_dir: Output directory
            file_type: GRIB2 file type ('wrfprs', 'wrfsfc', 'wrfnat')
        """
        cycle_dt = datetime.strptime(cycle, '%Y%m%d%H')
        date_str = cycle_dt.strftime('%Y%m%d')
        
        # Build filename based on file type
        filename = f'hrrr.t{cycle[-2:]}z.{file_type}f{forecast_hour:02d}.grib2'
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"File exists: {filename}")
            return output_path
        
        # Try sources in order: NOMADS (recent), AWS S3 (historical), Utah Pando (backup)
        # Based on guide: https://github.com/blaylockbk/Herbie
        pando_path = 'prs' if file_type == 'wrfprs' else 'sfc'
        sources = [
            f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}',
            f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/{filename}',
            f'https://pando-rgw01.chpc.utah.edu/hrrr/hrrr.{date_str}/conus/{filename}'
        ]
        
        for i, url in enumerate(sources):
            try:
                source_name = ["NOMADS", "AWS S3", "Utah Pando"][i]
                print(f"⬇️ Downloading {filename} from {source_name}...")
                
                # Set longer timeout for large files
                import socket
                socket.setdefaulttimeout(600)  # 10 minutes
                
                urllib.request.urlretrieve(url, output_path)
                print(f"✅ Downloaded from {source_name}: {filename}")
                return output_path
            except urllib.error.URLError as e:
                if i < len(sources) - 1:
                    next_source = ["NOMADS", "AWS S3", "Utah Pando"][i + 1]
                    print(f"⚠️ {source_name} failed ({e}), trying {next_source}...")
                else:
                    print(f"❌ {source_name} also failed ({e})")
                continue
        
        print(f"❌ Failed to download {filename} from all sources")
        return None

    def load_field_data(self, grib_file, field_name, field_config):
        """Load specific field data - now uses robust multi-dataset approach when needed"""
        # Check if this field requires robust loading
        if field_config.get('requires_multi_dataset'):
            return self.load_field_data_robust(grib_file, field_name, field_config)
        
        # Use original method for regular fields (keeps smoke working!)
        return self.load_field_data_original(grib_file, field_name, field_config)
    
    def load_field_data_original(self, grib_file, field_name, field_config):
        """Original single-dataset loading method - kept for reliable fields like smoke"""
        try:
            # Open dataset with specific access method
            access_keys = field_config['access'].copy()
            
            # For paramId-based fields, try to be more specific
            if 'paramId' in access_keys:
                # Add surface level specification for common surface fields
                surface_params = {167: 't2m', 168: 'd2m', 165: 'u10', 166: 'v10', 260242: 'r2'}
                if access_keys['paramId'] in surface_params:
                    access_keys['typeOfLevel'] = 'heightAboveGround'
                    access_keys['level'] = 2 if access_keys['paramId'] in [167, 168, 260242] else 10
            
            # Special handling for COLMD (column-integrated smoke)
            if field_config['var'] == 'COLMD_entireatmosphere_consideredasasinglelayer_':
                # Extract COLMD using wgrib2 and read as NetCDF
                import subprocess
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
                    temp_nc = tmp_file.name
                
                try:
                    # Find COLMD record
                    result = subprocess.run(['wgrib2', str(grib_file), '-s'], capture_output=True, text=True)
                    colmd_record = None
                    for line in result.stdout.strip().split('\n'):
                        if 'COLMD' in line:
                            colmd_record = line.split(':')[0]
                            break
                    
                    if colmd_record:
                        # Extract to NetCDF
                        subprocess.run(['wgrib2', str(grib_file), '-d', colmd_record, '-netcdf', temp_nc], 
                                     capture_output=True, text=True, check=True)
                        
                        # Load the NetCDF file
                        import xarray as xr
                        ds = xr.open_dataset(temp_nc)
                        print(f"🔍 COLMD variables available: {list(ds.data_vars.keys())}")
                    else:
                        print("❌ COLMD record not found")
                        return None
                        
                except Exception as e:
                    print(f"❌ Failed to extract COLMD: {e}")
                    return None
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_nc):
                        os.unlink(temp_nc)
            else:
                ds = cfgrib.open_dataset(grib_file, filter_by_keys=access_keys)
            
            if field_config['var'] not in ds.data_vars:
                print(f"❌ Variable {field_config['var']} not found")
                return None
            
            data = ds[field_config['var']]
            
            # Handle multi-dimensional data
            if field_config.get('process') == 'select_layer':
                # For SRH, select appropriate layer
                if 'heightAboveGroundLayer' in data.dims:
                    if len(data.heightAboveGroundLayer) > 1:
                        if '01km' in field_name:
                            data = data.isel(heightAboveGroundLayer=0)  # First layer
                        else:
                            data = data.isel(heightAboveGroundLayer=-1)  # Last layer
                    else:
                        data = data.isel(heightAboveGroundLayer=0)
            
            # Handle height-based reflectivity (select specific level)
            if 'heightAboveGround' in data.dims and len(data.dims) > 2:
                target_level = field_config['access'].get('level')
                if target_level:
                    # Find closest level
                    levels = data.heightAboveGround.values
                    closest_idx = np.argmin(np.abs(levels - target_level))
                    data = data.isel(heightAboveGround=closest_idx)
                    print(f"Selected height level: {levels[closest_idx]} m (target: {target_level} m)")
                else:
                    # Default to first level
                    data = data.isel(heightAboveGround=0)
                    print(f"Selected first available height level: {data.heightAboveGround.values} m")
            
            # Ensure data is 2D for plotting
            while len(data.dims) > 2:
                # Remove extra dimensions by selecting first index
                extra_dim = [dim for dim in data.dims if dim not in ['latitude', 'longitude', 'y', 'x']][0]
                data = data.isel({extra_dim: 0})
                print(f"Reduced dimension {extra_dim} to 2D")
            
            # Apply transformations
            if field_config.get('transform') == 'abs':
                data = abs(data)
            elif field_config.get('transform') == 'celsius':
                data = data - 273.15  # Kelvin to Celsius
            elif field_config.get('transform') == 'mb':
                data = data / 100  # Pa to mb
            elif field_config.get('transform') == 'smoke_concentration':
                # Convert from kg/m³ to μg/m³ (HRRR changed units in Dec 2021)
                data = data * 1e9  # kg/m³ to μg/m³
            elif field_config.get('transform') == 'smoke_column':
                # Convert column mass to mg/m²
                data = data * 1e6  # kg/m² to mg/m²
            elif field_config.get('transform') == 'dust_concentration':
                # Convert dust concentration from kg/m³ to μg/m³
                data = data * 1e9  # kg/m³ to μg/m³
            elif field_config.get('transform') == 'prate_units':
                # Convert precipitation rate from kg/m²/s to mm/hr
                data = data * 3600  # kg/m²/s to mm/hr
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to load {field_name}: {e}")
            return None
    
    def load_field_data_multids(self, grib_file, field_name, field_config):
        """Load field data with multi-dataset support - optimized version"""
        var_name = field_config['var']
        
        try:
            # Load all datasets with error suppression to avoid index warnings spam
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                datasets = cfgrib.open_datasets(grib_file, backend_kwargs={'indexpath': ''})
            
            print(f"🔍 Searching {var_name} across {len(datasets)} datasets...")
            
            # Search across all datasets for our variable
            for i, ds in enumerate(datasets):
                try:
                    if var_name in ds.data_vars:
                        print(f"✅ Found {var_name} in dataset {i}")
                        data = ds[var_name]
                        
                        # Apply layer selection if needed
                        if 'level_selection' in field_config:
                            data = self._select_layer(data, field_config['level_selection'])
                        
                        # Close other datasets to free memory
                        for j, other_ds in enumerate(datasets):
                            if j != i:
                                try:
                                    other_ds.close()
                                except:
                                    pass
                        
                        return data
                        
                    # Handle 'unknown' variables by checking GRIB metadata
                    if 'unknown' in ds.data_vars:
                        unknown_var = ds['unknown']
                        if hasattr(unknown_var, 'attrs'):
                            grib_name = unknown_var.attrs.get('GRIB_shortName', '')
                            grib_param_name = unknown_var.attrs.get('GRIB_parameterName', '')
                            
                            # Check both shortName and parameterName for matches
                            var_lower = var_name.lower()
                            grib_shortname_match = field_config.get('grib_shortname_match', '').lower()
                            
                            if (grib_name.lower() == var_lower or 
                                grib_param_name.lower().replace(' ', '_') == var_lower or
                                (grib_shortname_match and grib_name.lower() == grib_shortname_match)):
                                print(f"✅ Found {var_name} as 'unknown' in dataset {i} (GRIB: {grib_name})")
                                data = unknown_var
                                
                                # Apply layer selection if needed
                                if 'level_selection' in field_config:
                                    data = self._select_layer(data, field_config['level_selection'])
                                
                                # Close other datasets to free memory
                                for j, other_ds in enumerate(datasets):
                                    if j != i:
                                        try:
                                            other_ds.close()
                                        except:
                                            pass
                                
                                return data
                except Exception as ds_error:
                    # Skip problematic datasets
                    continue
            
            # Close all datasets
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass
            
            print(f"❌ Variable {var_name} not found in any dataset")
            return None
            
        except Exception as e:
            print(f"❌ Multi-dataset loading error: {e}")
            return None

    def _select_layer(self, data, layer_config):
        """Select specific layer from multi-dimensional data"""
        if 'heightAboveGroundLayer' in data.dims:
            layer_values = data.heightAboveGroundLayer.values
            print(f"🔍 Available layers: {layer_values}")
            
            if isinstance(layer_config, dict):
                bottom = layer_config.get('bottom', 0)
                top = layer_config.get('top', 3000)
                
                # Find matching layer - HRRR uses top value as identifier
                for i, layer_val in enumerate(layer_values):
                    if layer_val == top:
                        print(f"✅ Selected layer {top}m (index {i})")
                        return data.isel(heightAboveGroundLayer=i)
                
                # If exact match not found, try closest
                closest_idx = np.argmin(np.abs(layer_values - top))
                print(f"⚠️ Exact layer {top}m not found, using closest: {layer_values[closest_idx]}m")
                return data.isel(heightAboveGroundLayer=closest_idx)
                        
            elif isinstance(layer_config, int):
                # Direct layer index
                if layer_config < len(layer_values):
                    print(f"✅ Selected layer index {layer_config}: {layer_values[layer_config]}m")
                    return data.isel(heightAboveGroundLayer=layer_config)
                else:
                    print(f"⚠️ Layer index {layer_config} out of range, using last layer")
                    return data.isel(heightAboveGroundLayer=-1)
        
        return data

    def load_field_with_wgrib2(self, grib_file, field_config):
        """Use wgrib2 to extract specific records then load with cfgrib"""
        import subprocess
        import tempfile
        
        var_name = field_config['var']
        pattern = field_config.get('wgrib2_pattern', var_name)
        
        print(f"🔧 Using wgrib2 extraction for {var_name} with pattern: {pattern}")
        
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Extract specific record
            cmd = ['wgrib2', str(grib_file), '-match', pattern, '-grib', tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                print(f"✅ wgrib2 extracted record: {result.stdout.strip()}")
                
                # Load the isolated GRIB record and immediately read data into memory
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = cfgrib.open_dataset(tmp_path, backend_kwargs={'indexpath': ''})
                
                print(f"🔍 Available variables in extracted file: {list(ds.data_vars.keys())}")
                
                # Try different variable name mappings
                possible_names = [var_name, var_name.lower(), 'unknown']
                
                for name in possible_names:
                    if name in ds.data_vars:
                        print(f"✅ Successfully extracted {var_name} as '{name}' with wgrib2")
                        # Load data into memory immediately
                        data = ds[name].load()
                        ds.close()
                        return data
                
                # If no direct match, return the first variable
                if len(ds.data_vars) > 0:
                    first_var = list(ds.data_vars.keys())[0]
                    print(f"✅ Using first available variable '{first_var}' for {var_name}")
                    # Load data into memory immediately
                    data = ds[first_var].load()
                    ds.close()
                    return data
            else:
                print(f"❌ wgrib2 extraction failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ wgrib2 extraction error: {e}")
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
        return None

    def load_field_data_robust(self, grib_file, field_name, field_config):
        """Robust field loading with multiple strategies"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Loading operation timed out")
        
        # Strategy 1: wgrib2 extraction FIRST for multi-dataset fields (most reliable)
        if field_config.get('requires_multi_dataset') and field_config.get('wgrib2_pattern'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout for wgrib2
                
                result = self.load_field_with_wgrib2(grib_file, field_config)
                signal.alarm(0)  # Cancel timeout
                
                if result is not None:
                    print(f"✅ Loaded {field_name} with wgrib2 approach")
                    return self._apply_data_transformations(result, field_config)
            except (Exception, TimeoutError) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"⚠️ wgrib2 approach failed for {field_name}: {e}")
        
        # Strategy 2: Try original single-dataset approach for non-multi-dataset fields
        if not field_config.get('requires_multi_dataset'):
            try:
                # Set timeout for single-dataset approach
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                
                result = self.load_field_data_original(grib_file, field_name, field_config)
                signal.alarm(0)  # Cancel timeout
                
                if result is not None:
                    print(f"✅ Loaded {field_name} with single-dataset approach")
                    return self._apply_data_transformations(result, field_config)
            except (Exception, TimeoutError) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"⚠️ Single-dataset approach failed for {field_name}: {e}")
        
        # Strategy 3: Multi-dataset search (fallback for problematic cases)
        if field_config.get('requires_multi_dataset'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 second timeout for multi-dataset
                
                result = self.load_field_data_multids(grib_file, field_name, field_config)
                signal.alarm(0)  # Cancel timeout
                
                if result is not None:
                    print(f"✅ Loaded {field_name} with multi-dataset approach")
                    return self._apply_data_transformations(result, field_config)
            except (Exception, TimeoutError) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"⚠️ Multi-dataset approach failed for {field_name}: {e}")
        
        print(f"❌ All strategies failed for {field_name}")
        return None

    def _apply_data_transformations(self, data, field_config):
        """Apply transformations and ensure data is ready for plotting"""
        # Handle multi-dimensional data
        if field_config.get('process') == 'select_layer':
            # For SRH, select appropriate layer
            if 'heightAboveGroundLayer' in data.dims:
                if len(data.heightAboveGroundLayer) > 1:
                    if '01km' in field_config.get('var', ''):
                        data = data.isel(heightAboveGroundLayer=0)  # First layer
                    else:
                        data = data.isel(heightAboveGroundLayer=-1)  # Last layer
                else:
                    data = data.isel(heightAboveGroundLayer=0)
        
        # Handle height-based reflectivity (select specific level)
        if 'heightAboveGround' in data.dims and len(data.dims) > 2:
            target_level = field_config.get('access', {}).get('level')
            if target_level:
                # Find closest level
                levels = data.heightAboveGround.values
                closest_idx = np.argmin(np.abs(levels - target_level))
                data = data.isel(heightAboveGround=closest_idx)
                print(f"Selected height level: {levels[closest_idx]} m (target: {target_level} m)")
            else:
                # Default to first level
                data = data.isel(heightAboveGround=0)
                print(f"Selected first available height level: {data.heightAboveGround.values} m")
        
        # Ensure data is 2D for plotting
        while len(data.dims) > 2:
            # Remove extra dimensions by selecting first index
            extra_dim = [dim for dim in data.dims if dim not in ['latitude', 'longitude', 'y', 'x']][0]
            data = data.isel({extra_dim: 0})
            print(f"Reduced dimension {extra_dim} to 2D")
        
        # Apply transformations
        if field_config.get('transform') == 'abs':
            data = abs(data)
        elif field_config.get('transform') == 'celsius':
            data = data - 273.15  # Kelvin to Celsius
        elif field_config.get('transform') == 'mb':
            data = data / 100  # Pa to mb
        elif field_config.get('transform') == 'smoke_concentration':
            # Convert from kg/m³ to μg/m³ (HRRR changed units in Dec 2021)
            data = data * 1e9  # kg/m³ to μg/m³
        elif field_config.get('transform') == 'smoke_column':
            # Convert column mass to mg/m²
            data = data * 1e6  # kg/m² to mg/m²
        elif field_config.get('transform') == 'dust_concentration':
            # Convert dust concentration from kg/m³ to μg/m³
            data = data * 1e9  # kg/m³ to μg/m³
        elif field_config.get('transform') == 'prate_units':
            # Convert precipitation rate from kg/m²/s to mm/hr
            data = data * 3600  # kg/m²/s to mm/hr
        elif field_config.get('transform') == 'hail_size':
            # Convert hail diameter from m to mm 
            data = data * 1000  # m to mm
        
        return data

    def load_uh_layer(self, path, top, bottom):
        """
        Return max-1h UH for a given AG layer (m AGL) from a HRRR wrfsfc file.
        `top` > `bottom`, e.g. top=3000, bottom=0 for 0–3 km.
        
        Args:
            path: Path to HRRR wrfsfc file
            top: Top of layer (m AGL)
            bottom: Bottom of layer (m AGL)
            
        Returns:
            xarray.DataArray of max updraft helicity
        """
        try:
            import xarray as xr
            
            # Try loading with paramId first (more specific)
            param_ids = {
                (3000, 0): 237137,    # 0-3km MXUPHL 
                (5000, 2000): 237138, # 2-5km MXUPHL
                (2000, 0): 237139     # 0-2km MXUPHL
            }
            
            layer_key = (top, bottom)
            if layer_key in param_ids:
                try:
                    ds = xr.open_dataset(
                        path,
                        engine="cfgrib",
                        indexpath="",
                        filter_by_keys={
                            "paramId": param_ids[layer_key]
                        }
                    )
                    # Get the variable (should be single variable with paramId)
                    var_name = list(ds.data_vars)[0]
                    uh_data = ds[var_name]
                    
                    # Check if we loaded the wrong variable (max_vo instead of MXUPHL)
                    if var_name == 'max_vo':
                        print(f"⚠️ paramId loaded wrong variable '{var_name}' instead of MXUPHL, falling back to wgrib2...")
                        raise Exception("Wrong variable loaded from paramId, forcing wgrib2 fallback")
                    
                    # Check if data is all zeros BEFORE dimension processing (indicating wrong field loaded)
                    data_max = float(uh_data.max().values)
                    if data_max == 0.0:
                        print(f"⚠️ paramId loaded all-zero data (var: {var_name}), falling back to wgrib2...")
                        raise Exception("All-zero data from paramId, forcing wgrib2 fallback")
                    
                    # Ensure 2D data for plotting (squeeze out extra dims)
                    if len(uh_data.dims) > 2:
                        # Keep only spatial dimensions (y/x or latitude/longitude)
                        dims_to_keep = ['latitude', 'longitude', 'y', 'x']
                        for dim in uh_data.dims:
                            if dim not in dims_to_keep and uh_data.sizes[dim] == 1:
                                uh_data = uh_data.squeeze(dim)
                            elif dim not in dims_to_keep:
                                # For non-spatial dimensions with size > 1, take the first slice
                                uh_data = uh_data.isel({dim: 0})
                    
                    print(f"✅ Loaded UH {bottom}-{top}m layer via paramId from {path}")
                    print(f"   Data shape: {uh_data.shape}, dims: {uh_data.dims}")
                    print(f"   Data range: {uh_data.min().values:.1f} to {uh_data.max().values:.1f}")
                    return uh_data
                except Exception as e:
                    print(f"⚠️ paramId approach failed: {e}")
            
            # Fallback: Try using wgrib2 pattern
            wgrib2_patterns = {
                (3000, 0): "MXUPHL:3000-0 m above ground",
                (5000, 2000): "MXUPHL:5000-2000 m above ground", 
                (2000, 0): "MXUPHL:2000-0 m above ground"
            }
            
            if layer_key in wgrib2_patterns:
                try:
                    import subprocess
                    import tempfile
                    import os
                    
                    # Extract specific MXUPHL layer using wgrib2
                    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp:
                        cmd = [
                            'wgrib2', str(path), 
                            '-match', wgrib2_patterns[layer_key],
                            '-grib', tmp.name
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            # Load the extracted data
                            ds = xr.open_dataset(tmp.name, engine="cfgrib")
                            var_name = list(ds.data_vars)[0]
                            uh_data = ds[var_name]
                            
                            # Ensure 2D data for plotting (squeeze out extra dims)
                            if len(uh_data.dims) > 2:
                                # Keep only spatial dimensions (y/x or latitude/longitude)
                                dims_to_keep = ['latitude', 'longitude', 'y', 'x']
                                for dim in uh_data.dims:
                                    if dim not in dims_to_keep and uh_data.sizes[dim] == 1:
                                        uh_data = uh_data.squeeze(dim)
                                    elif dim not in dims_to_keep:
                                        # For non-spatial dimensions with size > 1, take the first slice
                                        uh_data = uh_data.isel({dim: 0})
                            
                            print(f"✅ Loaded UH {bottom}-{top}m layer via wgrib2 from {path}")
                            print(f"   Data shape: {uh_data.shape}, dims: {uh_data.dims}")
                            print(f"   Data range: {uh_data.min().values:.1f} to {uh_data.max().values:.1f}")
                            
                            # Clean up temp file
                            os.unlink(tmp.name)
                            return uh_data
                        else:
                            os.unlink(tmp.name)
                            print(f"⚠️ wgrib2 extraction failed: {result.stderr}")
                            
                except Exception as e:
                    print(f"⚠️ wgrib2 approach failed: {e}")
            
            # Final fallback - return None
            print(f"❌ All approaches failed for UH layer {bottom}-{top}m")
            return None
            
        except Exception as e:
            print(f"❌ Failed to load UH layer {bottom}-{top}m: {e}")
            return None

    def load_derived_parameter(self, field_name, field_config, grib_file, wrfsfc_file=None):
        """Load and compute derived parameter from input fields"""
        try:
            print(f"🧮 Computing derived parameter: {field_name}")
            
            # Get input field names and function
            input_fields = field_config.get('inputs', [])
            function_name = field_config.get('function')
            
            if not input_fields or not function_name:
                print(f"❌ Missing inputs or function for derived parameter {field_name}")
                return None
            
            # Load all input fields
            input_data = {}
            for input_field in input_fields:
                print(f"  Loading input field: {input_field}")
                
                # Get configuration for input field
                input_config = self.registry.get_field(input_field)
                if not input_config:
                    print(f"❌ Input field configuration not found: {input_field}")
                    return None
                
                # Check if input field is also derived (recursive)
                if input_config.get('derived'):
                    data = self.load_derived_parameter(input_field, input_config, grib_file, wrfsfc_file)
                else:
                    # Load regular field data
                    if input_config.get('category') == 'smoke' and wrfsfc_file:
                        data = self.load_field_data(wrfsfc_file, input_field, input_config)
                        if data is None:
                            data = self.load_field_data(grib_file, input_field, input_config)
                    else:
                        data = self.load_field_data(grib_file, input_field, input_config)
                
                if data is None:
                    print(f"❌ Failed to load input field: {input_field}")
                    return None
                
                # Convert to numpy array for computation
                input_data[input_field] = data.values
                print(f"  ✅ Loaded {input_field}: {data.shape}")
            
            # Note: SCP will use fallback recipe if mucin is unavailable
            
            # Compute derived parameter
            print(f"  Computing {function_name}...")
            result_array = compute_derived_parameter(field_name, input_data, field_config)
            
            if result_array is None:
                print(f"❌ Failed to compute derived parameter: {field_name}")
                return None
            
            # Create xarray DataArray with coordinates from one of the input fields
            # Use the first successfully loaded input field for coordinates
            reference_field = list(input_data.keys())[0]
            reference_config = self.registry.get_field(reference_field)
            
            # Load reference field as xarray to get coordinates
            if reference_config.get('derived'):
                # For derived reference fields, we need to find a non-derived input field for coordinates
                # Look for the first non-derived input in the reference field's inputs
                ref_inputs = reference_config.get('inputs', [])
                ref_data = None
                for ref_input in ref_inputs:
                    ref_input_config = self.registry.get_field(ref_input)
                    if not ref_input_config.get('derived'):
                        # Found a non-derived input, use it for coordinates
                        if ref_input_config.get('category') == 'smoke' and wrfsfc_file:
                            ref_data = self.load_field_data(wrfsfc_file, ref_input, ref_input_config)
                            if ref_data is None:
                                ref_data = self.load_field_data(grib_file, ref_input, ref_input_config)
                        else:
                            ref_data = self.load_field_data(grib_file, ref_input, ref_input_config)
                        if ref_data is not None:
                            break
                # If still no reference data, use the computed reference field itself
                if ref_data is None:
                    # Use one of the already loaded input fields for coordinates
                    for inp_name, inp_array in input_data.items():
                        inp_config = self.registry.get_field(inp_name)
                        if not inp_config.get('derived'):
                            if inp_config.get('category') == 'smoke' and wrfsfc_file:
                                ref_data = self.load_field_data(wrfsfc_file, inp_name, inp_config)
                                if ref_data is None:
                                    ref_data = self.load_field_data(grib_file, inp_name, inp_config)
                            else:
                                ref_data = self.load_field_data(grib_file, inp_name, inp_config)
                            if ref_data is not None:
                                break
            else:
                # Regular field, load normally
                if reference_config.get('category') == 'smoke' and wrfsfc_file:
                    ref_data = self.load_field_data(wrfsfc_file, reference_field, reference_config)
                    if ref_data is None:
                        ref_data = self.load_field_data(grib_file, reference_field, reference_config)
                else:
                    ref_data = self.load_field_data(grib_file, reference_field, reference_config)
            
            if ref_data is None:
                print(f"❌ Could not get reference coordinates")
                return None
            
            # Create DataArray with same coordinates as reference
            result_data = xr.DataArray(
                result_array,
                coords=ref_data.coords,
                dims=ref_data.dims,
                name=field_name,
                attrs={
                    'long_name': field_config.get('title', field_name),
                    'units': field_config.get('units', 'dimensionless'),
                    'derived': True
                }
            )
            
            print(f"✅ Successfully computed derived parameter: {field_name}")
            return result_data
            
        except Exception as e:
            print(f"❌ Error computing derived parameter {field_name}: {e}")
            return None

    def create_spc_plot(self, data, field_name, field_config, cycle, forecast_hour, output_dir):
        """Create enhanced SPC-style plot with comprehensive metadata"""
        if data is None:
            return None
        
        # Calculate native aspect ratio with space for info panel
        aspect_ratio = data.shape[1] / data.shape[0]
        fig_width = 16  # Wider to accommodate info panel
        fig_height = fig_width / aspect_ratio * 0.75  # Adjust for info space
        
        # Create figure with subplots for map and info panel
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        
        # Create main map axes (takes up 75% of width)
        ax_map = plt.subplot2grid((1, 4), (0, 0), colspan=3, projection=ccrs.PlateCarree())
        ax_map.set_facecolor('white')
        
        # Create info panel axes (takes up 25% of width)
        ax_info = plt.subplot2grid((1, 4), (0, 3))
        ax_info.axis('off')  # Turn off axes for info panel
        
        # Set extent to CONUS for map
        ax_map.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
        ax_map.patch.set_facecolor('white')
        
        # Add map features
        ax_map.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black', facecolor='none', zorder=2)
        ax_map.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='black', facecolor='none', zorder=2)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', facecolor='none', zorder=2)
        
        # Get coordinates
        if hasattr(data, 'longitude') and hasattr(data, 'latitude'):
            lons = data.longitude.values
            lats = data.latitude.values
        else:
            print(f"❌ No coordinates found for {field_name}")
            return None
        
        # Convert longitude
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)
        
        # Prepare data
        plot_data = data.values.copy()
        
        # Mask invalid data
        if field_config.get('category') == 'smoke':
            # For smoke fields, don't mask out low values - show everything above 0
            plot_data = np.ma.masked_where(
                (np.isnan(plot_data)) | 
                (plot_data <= -9999) | 
                (plot_data <= 0), 
                plot_data
            )
        else:
            # For other fields, mask below first level as usual
            plot_data = np.ma.masked_where(
                (np.isnan(plot_data)) | 
                (plot_data <= -9999) | 
                (plot_data < field_config['levels'][0]), 
                plot_data
            )
        
        # Get colormap
        cmap_name = field_config['cmap']
        if cmap_name in self.colormaps:
            cmap = self.colormaps[cmap_name]
        else:
            cmap = plt.cm.get_cmap(cmap_name)
        
        # Plot data - support filled contours, contour lines, and composite plots
        plot_style = field_config.get('plot_style', 'filled')
        
        if plot_style == 'composite':
            # Composite plot: base field (filled) + overlay field (lines)
            base_config = field_config['base_field']
            overlay_config = field_config['overlay_field']
            
            # Get base field data (MLCIN) - load it directly
            base_param = base_config['parameter']
            
            # Try to load MLCIN from the GRIB file
            try:
                # Get MLCIN field configuration
                from field_registry import FieldRegistry
                registry = FieldRegistry()
                registry.build_all_configs({})  # Build configs
                
                if base_param in registry.all_fields:
                    mlcin_config = registry.all_fields[base_param]
                    # Load MLCIN data using the same GRIB files
                    grib_files = [grib_file for grib_file in [pressure_grib_file, surface_grib_file] if grib_file and os.path.exists(grib_file)]
                    
                    base_data = None
                    for grib_file in grib_files:
                        try:
                            base_data = self.load_field_data(grib_file, base_param, mlcin_config)
                            if base_data is not None:
                                break
                        except:
                            continue
                    
                    if base_data is not None:
                        base_data = np.ma.masked_where(
                            (np.isnan(base_data)) | (base_data <= -9999), base_data
                        )
                        
                        # Plot base field (filled contours)
                        base_cmap = plt.cm.get_cmap(base_config['cmap'])
                        cs_base = ax_map.contourf(lons, lats, base_data,
                                                 levels=base_config['levels'],
                                                 cmap=base_cmap,
                                                 extend=base_config['extend'],
                                                 transform=ccrs.PlateCarree(),
                                                 zorder=1)
                        
                        # Add colorbar for base field
                        cbar = plt.colorbar(cs_base, ax=ax_map, orientation='horizontal', 
                                           pad=0.05, shrink=0.8, aspect=30)
                        cbar.set_label(f"{base_config['title']} ({base_config['units']})", 
                                       fontsize=12, fontweight='bold')
                        cbar.ax.tick_params(labelsize=10)
            except Exception as e:
                print(f"⚠️ Could not load base field {base_param}: {e}")
                # Fall back to no base field
            
            # Plot overlay field (VTP contour lines)
            overlay_data = plot_data  # VTP data
            overlay_data = np.ma.masked_where(
                (np.isnan(overlay_data)) | (overlay_data <= 0), overlay_data
            )
            
            colors = overlay_config.get('colors', ['red'])
            linewidths = overlay_config.get('linewidths', [1.5])
            
            cs_overlay = ax_map.contour(lons, lats, overlay_data,
                                       levels=overlay_config['levels'],
                                       colors=colors,
                                       linewidths=linewidths,
                                       transform=ccrs.PlateCarree(),
                                       zorder=3)
            # Add contour labels for overlay
            ax_map.clabel(cs_overlay, inline=True, fontsize=9, fmt='%.1f')
            
        elif plot_style == 'spc_vtp':
            # SPC-style VTP panel: MLCIN shading + dashed CIN isolines + VTP contours
            import matplotlib.colors as mcolors
            
            spc_config = field_config.get('spc_config', {})
            
            # Load MLCIN data for base shading
            try:
                from field_registry import FieldRegistry
                registry = FieldRegistry()
                registry.build_all_configs({})
                
                if 'mlcin' in registry.all_fields:
                    mlcin_config = registry.all_fields['mlcin']
                    # Try to load MLCIN from available GRIB files
                    mlcin_data = None
                    grib_files = [f for f in [pressure_grib_file, surface_grib_file] if f and os.path.exists(f)]
                    
                    for grib_file in grib_files:
                        try:
                            mlcin_data = self.load_field_data(grib_file, 'mlcin', mlcin_config)
                            if mlcin_data is not None:
                                break
                        except:
                            continue
                    
                    if mlcin_data is not None:
                        # 1. MLCIN shading (cyan with hatching)
                        cin_shade_levels = spc_config.get('cin_shade_levels', [-100, -25, 0])
                        cin_colors = spc_config.get('cin_cmap_colors', ['#00d5ff', '#b0f0ff'])
                        cin_hatches = spc_config.get('cin_hatches', ['////', None])
                        
                        cin_cmap = mcolors.ListedColormap(cin_colors)
                        cin_masked = np.ma.masked_where(mlcin_data > -25, mlcin_data)
                        
                        cf = ax_map.contourf(lons, lats, cin_masked,
                                           levels=cin_shade_levels,
                                           cmap=cin_cmap, extend='min',
                                           transform=ccrs.PlateCarree(),
                                           zorder=1)
                        
                        # Add hatching to collections
                        for i, coll in enumerate(cf.collections):
                            if i < len(cin_hatches) and cin_hatches[i]:
                                coll.set_hatch(cin_hatches[i])
                                coll.set_edgecolor('none')
                        
                        # 2. Dashed CIN isolines
                        cin_line_levels = spc_config.get('cin_line_levels', [-50, -25])
                        cin_line_color = spc_config.get('cin_line_color', '#d67800')
                        
                        ax_map.contour(lons, lats, mlcin_data,
                                     levels=cin_line_levels,
                                     colors=cin_line_color,
                                     linewidths=1.0,
                                     linestyles='--',
                                     transform=ccrs.PlateCarree(),
                                     zorder=3)
                        
                        # Add CIN colorbar (small, lower placement)
                        cbar = plt.colorbar(cf, ax=ax_map, orientation='horizontal',
                                          shrink=0.4, pad=0.03)
                        cbar.set_label('MLCIN (J kg⁻¹)', fontsize=10, fontweight='bold')
                        cbar.ax.tick_params(labelsize=8)
                    
            except Exception as e:
                print(f"⚠️ Could not load MLCIN for SPC-style plot: {e}")
            
            # 3. VTP contours (red → purple progression)
            vtp_levels = spc_config.get('vtp_levels', [2, 3, 4, 6, 8, 10, 12])
            vtp_colors = spc_config.get('vtp_colors', ['#ff0000']*4 + ['#9900ff']*3)
            vtp_linewidths = spc_config.get('vtp_linewidths', [1, 1, 1, 1.5, 2, 2.5, 3])
            
            # Mask VTP data to only show >= 2
            vtp_masked = np.ma.masked_where(plot_data < 2, plot_data)
            
            cs_vtp = ax_map.contour(lons, lats, vtp_masked,
                                   levels=vtp_levels,
                                   colors=vtp_colors,
                                   linewidths=vtp_linewidths,
                                   transform=ccrs.PlateCarree(),
                                   zorder=4)
            
            # Add VTP labels
            ax_map.clabel(cs_vtp, inline=True, fontsize=9, fmt='%.0f')
            
        elif plot_style == 'spc_style':
            # SPC-style filled contours with proper boundary norm (Hampshire et al. 2018 standards)
            import matplotlib.colors as mcolors
            levels = field_config['levels']
            cmap = plt.cm.get_cmap('plasma', len(levels)-1)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            
            # Mask data below threshold
            plot_data = np.ma.masked_where((plot_data < levels[0]) | (np.isnan(plot_data)), plot_data)
            
            cs = ax_map.contourf(lons, lats, plot_data,
                                levels=levels, cmap=cmap, norm=norm,
                                extend=field_config['extend'],
                                transform=ccrs.PlateCarree(),
                                zorder=1)
            
            # Add red contour lines for VTP >= 2 (SPC overlay style)
            high_levels = [l for l in levels if l >= 2]
            if len(high_levels) > 0:
                cs_lines = ax_map.contour(lons, lats, plot_data,
                                         levels=high_levels,
                                         colors='red', linewidths=1.5,
                                         transform=ccrs.PlateCarree(),
                                         zorder=2)
                ax_map.clabel(cs_lines, inline=True, fontsize=9, fmt='%.0f')
            
        elif plot_style == 'multicolor_lines':
            # Multi-colored contour lines (like SPC style)
            colors = field_config.get('line_colors', ['red'])
            widths = field_config.get('line_widths', [1.5])
            
            cs = ax_map.contour(lons, lats, plot_data,
                               levels=field_config['levels'],
                               colors=colors,
                               linewidths=widths,
                               transform=ccrs.PlateCarree(),
                               zorder=2)
            # Add contour labels
            ax_map.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
            
        elif plot_style == 'lines':
            # Use contour lines for parameters like VTP
            cs = ax_map.contour(lons, lats, plot_data,
                               levels=field_config['levels'],
                               colors='red',
                               linewidths=1.5,
                               transform=ccrs.PlateCarree(),
                               zorder=2)
            # Add contour labels
            ax_map.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
        else:
            # Default filled contours
            cs = ax_map.contourf(lons, lats, plot_data,
                                 levels=field_config['levels'],
                                 cmap=cmap,
                                 extend=field_config['extend'],
                                 transform=ccrs.PlateCarree(),
                                 zorder=1)
        
        # Add colorbar (for filled contours and SPC style, but not spc_vtp which handles its own)
        if plot_style in ['filled', 'spc_style']:
            cbar = plt.colorbar(cs, ax=ax_map, orientation='horizontal', 
                               pad=0.05, shrink=0.8, aspect=30)
            cbar.set_label(f"{field_config['title']} ({field_config['units']})", 
                           fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
        
        # Enhanced title for main map
        cycle_dt = datetime.strptime(cycle, '%Y%m%d%H')
        valid_dt = cycle_dt + timedelta(hours=forecast_hour)
        
        ax_map.set_title(f"HRRR {field_config['title']}\n"
                        f"Valid: {valid_dt.strftime('%Y-%m-%d %H UTC')} | F{forecast_hour:02d}",
                        fontsize=14, fontweight='bold', pad=15)
        
        # Create comprehensive info panel
        info_text = self._create_info_panel(field_config, cycle_dt, valid_dt, forecast_hour, data)
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Save
        output_file = output_dir / f"{field_name}_f{forecast_hour:02d}_REFACTORED.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        return output_file
    
    def _create_info_panel(self, field_config, cycle_dt, valid_dt, forecast_hour, data):
        """Create comprehensive information panel text"""
        
        # Basic timing info
        info_lines = [
            "═══ HRRR METADATA ═══",
            f"Model: High-Resolution Rapid Refresh",
            f"Init: {cycle_dt.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Valid: {valid_dt.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Forecast Hour: F{forecast_hour:02d}",
            "",
            "═══ PARAMETER INFO ═══",
            f"Name: {field_config.get('title', 'Unknown')}",
            f"Units: {field_config.get('units', 'dimensionless')}",
            f"Category: {field_config.get('category', 'general').title()}",
        ]
        
        # Add derived parameter info if applicable
        if field_config.get('derived'):
            info_lines.extend([
                f"Type: Derived Parameter",
                f"Function: {field_config.get('function', 'unknown')}",
            ])
            if field_config.get('inputs'):
                inputs_str = ', '.join(field_config['inputs'])
                info_lines.append(f"Inputs: {inputs_str}")
            
            # Add actual calculation formula
            formula = self._extract_formula(field_config.get('function'))
            if formula:
                info_lines.extend([
                    "",
                    "═══ CALCULATION ═══",
                    f"{formula}"
                ])
        else:
            info_lines.append("Type: Direct GRIB Field")
        
        # Add description if available
        if field_config.get('description'):
            desc = field_config['description']
            # Wrap long descriptions
            if len(desc) > 40:
                desc = desc[:37] + "..."
            info_lines.extend(["", f"Description:", f"{desc}"])
        
        # Data statistics
        if data is not None:
            try:
                valid_data = data.values[~np.isnan(data.values)]
                if len(valid_data) > 0:
                    info_lines.extend([
                        "",
                        "═══ DATA STATS ═══",
                        f"Min: {valid_data.min():.2f}",
                        f"Max: {valid_data.max():.2f}",
                        f"Mean: {valid_data.mean():.2f}",
                        f"Grid: {data.shape[0]}×{data.shape[1]}",
                    ])
            except:
                pass
        
        # Color scale info
        if field_config.get('levels'):
            levels = field_config['levels']
            info_lines.extend([
                "",
                "═══ COLOR SCALE ═══",
                f"Levels: {len(levels)} intervals",
                f"Range: {levels[0]} to {levels[-1]}",
                f"Colormap: {field_config.get('cmap', 'default')}",
            ])
        
        # Add processing timestamp
        from datetime import datetime
        info_lines.extend([
            "",
            "═══ PROCESSING ═══",
            f"Generated: {datetime.utcnow().strftime('%H:%M:%S UTC')}",
            f"System: HRRR Processor v2",
        ])
        
        return '\n'.join(info_lines)
    
    def _extract_formula(self, function_name):
        """Extract the main calculation formula from a derived parameter function"""
        if not function_name:
            return None
            
        try:
            from derived_params import DerivedParameters
            import inspect
            import re
            
            # Get the function
            if hasattr(DerivedParameters, function_name):
                func = getattr(DerivedParameters, function_name)
                source = inspect.getsource(func)
                
                # Extract key calculation lines
                formula = self._parse_calculation_from_source(source, function_name)
                return formula
                
        except Exception as e:
            return f"Formula extraction error: {str(e)}"
        
        return None
    
    def _parse_calculation_from_source(self, source, function_name):
        """Parse source code to extract the main calculation"""
        import re
        
        # Known formula patterns for common calculations
        formulas = {
            'wbgt_shade': 'WBGT = 0.7 × WB + 0.3 × DB',
            'wbgt_estimated_outdoor': 'WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB\nBG = DB + solar - wind_cooling',
            'wbgt_simplified_outdoor': 'WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB\nBG = DB + 2°C - wind_cooling',
            'wet_bulb_temperature_metpy': 'Wet Bulb = f(T, Td, P)\nStull approximation if MetPy unavailable',
            'wind_speed_10m': 'Speed = √(u² + v²)',
            'wind_direction_10m': 'Direction = atan2(v, u) × 180/π',
            'mixing_ratio_2m': 'MR = 0.622 × es / (P - es)\nes = 6.112 × exp(17.67×Td/(Td+243.5))',
            'supercell_composite_parameter': 'SCP = (muCAPE/1000) × (ESRH/50) × clip((EBWD-10)/10, 0, 1) × CIN_term\nCIN_term = 1 if muCIN > -40, else -40/muCIN',
            'significant_tornado_parameter': 'STP = (MLCAPE/1500) × (2000-LCL)/1000\n    × (SRH/150) × (Shear/20)',
            'energy_helicity_index': 'EHI = (CAPE × SRH) / 160000',
            'bulk_richardson_number': 'BRN = CAPE / (0.5 × Shear²)',
            'crosswind_component': 'Crosswind = u × sin(θ) + v × cos(θ)\nwhere θ = reference_direction',
            'fire_weather_index': 'FWI = f(T, RH, WindSpeed)\nCombines temperature, humidity, wind',
            'wind_shear_magnitude': 'Shear = √(u_shear² + v_shear²)',
            'ventilation_rate_from_components': 'VR = WindSpeed × PBL_Height\nWindSpeed = √(u² + v²)',
            'effective_srh': 'Effective SRH = SRH × (CAPE/2500)\nwith LCL and CIN adjustments',
            'craven_brooks_composite': 'CBC = √((CAPE/2500) × (SRH/150) × (Shear/20))',
            'modified_stp_effective': 'Modified STP = (MLCAPE/1500) × (2000-LCL)/1000\n× (SRH/150) × (Shear/20) × CIN_factor',
            'surface_richardson_number': 'Ri = (g/T) × (dT/dz) / (du/dz)²\nStability parameter',
            'cross_totals': 'CT = Dewpoint_850 - Temperature_500\nInstability index',
            'violent_tornado_parameter': 'VTP = (MLCAPE/1500) × (EBWD/20) × (ESRH/150) × ((2000-MLLCL)/1000)\n    × ((200+MLCIN)/150) × (0-3km CAPE/50) × (0-3km Lapse/6.5)',
            'significant_tornado_parameter_cin': 'STP-CIN = (MLCAPE/1500) × (ESRH/150) × (EBWD/12)\n    × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)'
        }
        
        # Return known formula if available
        if function_name in formulas:
            return formulas[function_name]
        
        # Try to extract from source code patterns
        lines = source.split('\n')
        calculation_lines = []
        
        # Look for key calculation patterns
        for line in lines:
            line = line.strip()
            
            # Skip comments and docstrings
            if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                continue
                
            # Look for return statements with calculations
            if 'return ' in line and any(op in line for op in ['+', '-', '*', '/', '**', 'np.']):
                # Clean up the return statement
                formula_line = line.replace('return ', '').strip()
                # Remove numpy prefixes for readability
                formula_line = re.sub(r'np\.', '', formula_line)
                calculation_lines.append(f"= {formula_line}")
                
            # Look for key assignment lines with mathematical operations
            elif any(op in line for op in ['=', '+', '-', '*', '/', '**']) and any(keyword in line.lower() for keyword in ['temp', 'cape', 'shear', 'wind', 'wbgt', 'wet_bulb', 'ratio']):
                # Skip simple assignments like variable declarations
                if '=' in line and any(op in line for op in ['+', '-', '*', '/', '**']):
                    # Clean up the line
                    clean_line = re.sub(r'^\s*\w+\s*=\s*', '', line)
                    clean_line = re.sub(r'np\.', '', clean_line)
                    if len(clean_line) < 100:  # Only include reasonably short lines
                        calculation_lines.append(clean_line)
        
        # Return the most relevant calculation lines
        if calculation_lines:
            return '\n'.join(calculation_lines[:3])  # Limit to 3 lines max
        
        # Fallback: return a generic description
        return f"Complex calculation in {function_name}()\nSee source code for details"

    def process_fields(self, fields_to_process=None, cycle=None, forecast_hour=1, output_dir=None):
        """Process specified fields or all available fields
        
        Args:
            fields_to_process: List of field names, category name, or None for all
            cycle: HRRR cycle (YYYYMMDDHH format)
            forecast_hour: Forecast hour
            output_dir: Output directory
        """
        # Load field configurations
        all_fields = self.registry.load_all_fields()
        if not all_fields:
            print("❌ No field configurations available")
            return
        
        # Determine which fields to process
        if fields_to_process is None:
            # Process all fields
            fields = all_fields
        elif isinstance(fields_to_process, str):
            # Check if it's a category name
            if fields_to_process in self.registry.get_available_categories():
                fields = self.registry.get_fields_by_category(fields_to_process)
                print(f"🎯 Processing {len(fields)} fields from category: {fields_to_process}")
            else:
                # Single field name
                field_config = self.registry.get_field(fields_to_process)
                if field_config:
                    fields = {fields_to_process: field_config}
                else:
                    print(f"❌ Field not found: {fields_to_process}")
                    return
        elif isinstance(fields_to_process, list):
            # List of field names
            fields = {}
            for field_name in fields_to_process:
                field_config = self.registry.get_field(field_name)
                if field_config:
                    fields[field_name] = field_config
                else:
                    print(f"⚠️ Field not found: {field_name}")
        else:
            print("❌ Invalid fields_to_process parameter")
            return
        
        # Setup cycle and output directory
        if cycle is None:
            # Use recent cycle
            now = datetime.utcnow()
            cycle = (now - timedelta(hours=2)).strftime('%Y%m%d%H')
        
        if output_dir is None:
            output_dir = Path('./hrrr_output')
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Download HRRR files (prioritize wrfprs, but also try wrfsfc for smoke)
        print(f"🚀 Processing {len(fields)} fields for cycle {cycle} F{forecast_hour:02d}")
        
        # Check if we need additional files (smoke or updraft helicity)
        smoke_fields = [name for name, config in fields.items()
                        if config.get('category') == 'smoke']
        has_smoke = len(smoke_fields) > 0

        needs_uh = any(cfg.get('category') == 'updraft_helicity' for cfg in fields.values())
        
        # Check if using local GRIB files (from forecast hour directory)
        if hasattr(self, 'use_local_grib') and self.use_local_grib:
            # Look for GRIB files - they should be 2 levels up from category folder
            # Path: outputs/hrrr/20250603/22z/F00/F00/surface/ -> outputs/hrrr/20250603/22z/F00/
            grib_search_dir = output_dir.parent.parent if output_dir.parent.parent != output_dir.parent else output_dir.parent
            hour = cycle[-2:]
            
            # Look for existing GRIB files
            wrfprs_file = grib_search_dir / f'hrrr.t{hour}z.wrfprsf{forecast_hour:02d}.grib2'
            wrfsfc_file = grib_search_dir / f'hrrr.t{hour}z.wrfsfcf{forecast_hour:02d}.grib2'
            wrfsubh_file = grib_search_dir / f'hrrr.t{hour}z.wrfsubhf{forecast_hour:02d}.grib2'
            
            print(f"🔍 Looking for GRIB files in: {grib_search_dir}")
            print(f"🔍 wrfprs path: {wrfprs_file}")
            print(f"🔍 wrfsfc path: {wrfsfc_file}")
            
            if wrfprs_file.exists():
                grib_file = wrfprs_file
                print(f"📁 Using existing GRIB file: {grib_file}")
            else:
                print(f"⚠️ GRIB file not found at {wrfprs_file}, downloading...")
                grib_file = self.download_hrrr_file(cycle, forecast_hour, output_dir, 'wrfprs')
            
            if has_smoke or needs_uh:
                if wrfsfc_file.exists():
                    print(f"📁 Using existing wrfsfc file: {wrfsfc_file}")
                else:
                    print(f"⚠️ wrfsfc file not found at {wrfsfc_file}, downloading...")
                    wrfsfc_file = self.download_hrrr_file(cycle, forecast_hour, output_dir, 'wrfsfc')
        else:
            # Original download behavior
            grib_file = self.download_hrrr_file(cycle, forecast_hour, output_dir, 'wrfprs')
            
            # Also download wrfsfc if needed for smoke or updraft helicity
            wrfsfc_file = None
            if has_smoke or needs_uh:
                wrfsfc_file = self.download_hrrr_file(cycle, forecast_hour, output_dir, 'wrfsfc')
                if wrfsfc_file:
                    print(f"📊 Also downloaded wrfsfc file for auxiliary data")
        
        if not grib_file or not grib_file.exists():
            print("❌ Could not download HRRR file")
            return
        
        # Process each field
        success_count = 0
        failed_fields = []
        
        # Track timing between parameters to catch mysterious gaps
        last_parameter_end = time.time()
        
        for field_name, field_config in fields.items():
            # Add inter-parameter timing logging with aggressive debugging
            current_time = time.time()
            inter_param_gap = current_time - last_parameter_end
            
            # Log ALL gaps for debugging (even small ones)
            gap_info = f"🕐 Gap before {field_name}: {inter_param_gap:.3f}s (last_end: {last_parameter_end:.3f}, now: {current_time:.3f})"
            print(gap_info)
            
            if inter_param_gap > 5.0:  # Log gaps longer than 5 seconds
                print(f"⏱️  TIMING GAP DETECTED: {inter_param_gap:.1f}s gap before {field_name}")
                
                # Log the gap to the timing file
                gap_log = {
                    'event': 'inter_parameter_gap',
                    'gap_duration_seconds': round(inter_param_gap, 2),
                    'before_parameter': field_name,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'cycle': cycle,
                    'forecast_hour': forecast_hour
                }
                
                timing_file = output_dir / f"timing_results_{cycle}_F{forecast_hour:02d}.json"
                try:
                    if timing_file.exists():
                        with open(timing_file, 'r') as f:
                            all_timings = json.load(f)
                    else:
                        all_timings = []
                    all_timings.append(gap_log)
                    with open(timing_file, 'w') as f:
                        json.dump(all_timings, f, indent=2)
                    print(f"✅ Gap logged to {timing_file}")
                except Exception as timing_error:
                    print(f"⚠️ Failed to log gap timing: {timing_error}")
            
            print(f"\\n🧪 Processing: {field_name} [Category: {field_config.get('category', 'unknown')}]")
            start_time = time.time()
            
            # Add field processing timeout (especially for problematic fields like mllcl_height)
            field_timeout = 30  # seconds
            data = None
            
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Field {field_name} timed out after {field_timeout} seconds")
                
                # Set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(field_timeout)
                
                # Check if this is a derived parameter
                if field_config.get('derived'):
                    # Compute derived parameter
                    data = self.load_derived_parameter(field_name, field_config, grib_file, wrfsfc_file)
                else:
                    # Choose appropriate file for this field
                    if field_config.get('category') == 'smoke' and wrfsfc_file:
                        # Try wrfsfc first for smoke fields, fallback to wrfprs
                        data = self.load_field_data(wrfsfc_file, field_name, field_config)
                        if data is None:
                            print(f"⚠️ Smoke field {field_name} not found in wrfsfc, trying wrfprs...")
                            data = self.load_field_data(grib_file, field_name, field_config)
                    elif field_config.get('category') == 'updraft_helicity' and wrfsfc_file:
                        # Check if field has layer specification for direct UH loading
                        if 'layer' in field_config:
                            top, bottom = map(int, field_config['layer'].split('-'))
                            data = self.load_uh_layer(wrfsfc_file, top, bottom)
                        else:
                            # Fallback to standard loading for UH fields without layer spec
                            data = self.load_field_data(wrfsfc_file, field_name, field_config)
                        
                        if data is None:
                            print(f"⚠️ UH field {field_name} not found in wrfsfc, trying wrfprs...")
                            data = self.load_field_data(grib_file, field_name, field_config)
                    else:
                        # Use wrfprs for all other fields
                        data = self.load_field_data(grib_file, field_name, field_config)
                
                # Clear timeout
                signal.alarm(0)
                
            except TimeoutError as e:
                print(f"⏰ TIMEOUT: {e}")
                data = None
            except Exception as e:
                print(f"❌ ERROR: {field_name} failed: {e}")
                data = None
            finally:
                # Ensure timeout is cleared
                try:
                    signal.alarm(0)
                except:
                    pass
            
            if data is not None:
                # Create plot
                output_file = self.create_spc_plot(data, field_name, field_config, 
                                                 cycle, forecast_hour, output_dir)
                
                duration = time.time() - start_time
                if output_file:
                    print(f"✅ SUCCESS: {output_file.name} ({duration:.1f}s)")
                    success_count += 1
                    
                    # Log timing for each parameter
                    timing_log = {
                        'parameter': field_name,
                        'category': field_config.get('category', 'unknown'),
                        'duration_seconds': round(duration, 2),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'cycle': cycle,
                        'forecast_hour': forecast_hour
                    }
                    
                    # Write timing to JSON file
                    timing_file = output_dir / f"timing_results_{cycle}_F{forecast_hour:02d}.json"
                    try:
                        if timing_file.exists():
                            with open(timing_file, 'r') as f:
                                all_timings = json.load(f)
                        else:
                            all_timings = []
                        all_timings.append(timing_log)
                        with open(timing_file, 'w') as f:
                            json.dump(all_timings, f, indent=2)
                    except Exception as timing_error:
                        print(f"⚠️ Failed to log timing: {timing_error}")
                        
                else:
                    duration = time.time() - start_time
                    print(f"❌ PLOT FAILED: {field_name} ({duration:.1f}s)")
                    failed_fields.append(field_name)
            else:
                duration = time.time() - start_time
                print(f"❌ DATA FAILED: {field_name} ({duration:.1f}s)")
                failed_fields.append(field_name)
            
            # Update timing for next iteration
            last_parameter_end = time.time()
            print(f"📝 {field_name} completed at {last_parameter_end:.3f} ({time.strftime('%H:%M:%S')})")
        
        # Summary
        print(f"\\n" + "="*60)
        print(f"📊 PROCESSING RESULTS")
        print(f"="*60)
        print(f"✅ SUCCESS: {success_count}/{len(fields)} ({success_count/len(fields)*100:.1f}%)")
        print(f"❌ FAILED: {len(failed_fields)} fields")
        
        if failed_fields:
            print(f"\\n❌ Failed fields:")
            for field in failed_fields:
                print(f"  - {field}")
        
        print(f"\\n📁 Output directory: {output_dir}")

    def list_available_fields(self, category=None):
        """List available fields, optionally by category"""
        if category:
            fields = self.registry.get_fields_by_category(category)
            print(f"\\n📋 Available fields in category '{category}':")
        else:
            fields = self.registry.get_all_fields()
            print(f"\\n📋 All available fields:")
        
        # Group by category for display
        categories = {}
        for field_name, config in fields.items():
            cat = config.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(field_name)
        
        for cat, field_list in sorted(categories.items()):
            print(f"\\n  {cat.upper()}:")
            for field in sorted(field_list):
                config = fields[field]
                print(f"    {field}: {config['title']} ({config['units']})")

    def add_custom_field(self, field_name, template_name, overrides=None, save=False):
        """Add a custom field using a template
        
        Args:
            field_name: Name for the new field
            template_name: Base template to use
            overrides: Dictionary of values to override from template
            save: Whether to save to configuration file
        """
        field_config = {'template': template_name}
        if overrides:
            field_config.update(overrides)
        
        success = self.registry.add_field(field_name, field_config, save_to_file=save)
        if success:
            print(f"✅ Added custom field: {field_name}")
            # Show the resulting configuration
            config = self.registry.get_field(field_name)
            print(f"   Template: {template_name}")
            print(f"   Title: {config['title']}")
            print(f"   Units: {config['units']}")
            if save:
                print(f"   💾 Saved to configuration file")
        
        return success


def main():
    """Main function with examples of using the refactored processor"""
    processor = HRRRProcessor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            # List available fields
            if len(sys.argv) > 2:
                processor.list_available_fields(sys.argv[2])
            else:
                processor.list_available_fields()
                
        elif command == 'categories':
            # List available categories
            categories = processor.registry.get_available_categories()
            print(f"\\n📂 Available categories: {', '.join(categories)}")
            
        elif command == 'process':
            # Process fields
            if len(sys.argv) > 2:
                fields_arg = sys.argv[2]
                # Check if it's a category or field name
                processor.process_fields(fields_to_process=fields_arg)
            else:
                # Process all fields (test mode - first 5 fields only)
                all_fields = processor.registry.get_all_fields()
                test_fields = list(all_fields.keys())[:5]
                print(f"🧪 Test mode: processing first 5 fields")
                processor.process_fields(fields_to_process=test_fields)
                
        elif command == 'add':
            # Add custom field example
            if len(sys.argv) > 3:
                field_name = sys.argv[2]
                template_name = sys.argv[3]
                processor.add_custom_field(field_name, template_name, save=True)
            else:
                print("Usage: python hrrr_processor_refactored.py add <field_name> <template_name>")
                
        elif command == 'search':
            # Search fields
            if len(sys.argv) > 2:
                search_term = sys.argv[2]
                results = processor.registry.search_fields(search_term)
                print(f"\\n🔍 Fields matching '{search_term}': {results}")
            else:
                print("Usage: python hrrr_processor_refactored.py search <term>")
                
        else:
            print("Unknown command. Available: list, categories, process, add, search")
    else:
        # Default: show available options
        print("🚀 HRRR Processor - Refactored Extensible Version")
        print("="*60)
        print("Usage:")
        print("  list [category]     - List available fields")
        print("  categories          - List available categories") 
        print("  process [field/cat] - Process fields")
        print("  add <name> <template> - Add custom field")
        print("  search <term>       - Search fields")
        print("\\nExamples:")
        print("  python hrrr_processor_refactored.py list instability")
        print("  python hrrr_processor_refactored.py process sbcape")
        print("  python hrrr_processor_refactored.py process instability")
        print("  python hrrr_processor_refactored.py add my_cape surface_cape")


if __name__ == '__main__':
    main()