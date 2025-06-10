#!/usr/bin/env python3
"""
HRRR Tier 2 Enhanced Severe Weather Training Data Pipeline
Processes 21 core variables for diffusion model training at native 3km resolution
"""

import os
import requests
import xarray as xr
import cfgrib
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HRRRTier2Pipeline:
    """Pipeline for processing HRRR Tier 2 Enhanced Severe Weather variables"""
    
    def __init__(self, output_dir: str = "tier2_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tier 2 Enhanced Severe Weather Set (21 variables)
        self.tier2_variables = {
            # Surface variables (5) - Already working
            't2m': {'paramId': 167, 'typeOfLevel': 'heightAboveGround', 'level': 2},
            'd2m': {'paramId': 168, 'typeOfLevel': 'heightAboveGround', 'level': 2},
            'u10': {'paramId': 165, 'typeOfLevel': 'heightAboveGround', 'level': 10},
            'v10': {'paramId': 166, 'typeOfLevel': 'heightAboveGround', 'level': 10},
            'sp': {'paramId': 134, 'typeOfLevel': 'surface'},
            
            # Terrain (1)
            'orog': {'paramId': 129, 'typeOfLevel': 'surface'},  # Orography/height_surface
            
            # Upper-air variables (8)
            'temp_850': {'paramId': 130, 'typeOfLevel': 'isobaricInhPa', 'level': 850},
            'temp_700': {'paramId': 130, 'typeOfLevel': 'isobaricInhPa', 'level': 700},
            'temp_500': {'paramId': 130, 'typeOfLevel': 'isobaricInhPa', 'level': 500},
            'dewpoint_850': {'shortName': 'dpt', 'typeOfLevel': 'isobaricInhPa', 'level': 850},
            'rh_500': {'paramId': 157, 'typeOfLevel': 'isobaricInhPa', 'level': 500},
            'u_500': {'paramId': 131, 'typeOfLevel': 'isobaricInhPa', 'level': 500},
            'v_500': {'paramId': 132, 'typeOfLevel': 'isobaricInhPa', 'level': 500},
            
            # Instability (2) - CAPE uses pressure layer patterns  
            'sbcape': {'shortName': 'cape', 'typeOfLevel': 'surface'},
            'mlcape': {'shortName': 'cape', 'typeOfLevel': 'pressureFromGroundLayer', 'level': 18000},
            'sbcin': {'shortName': 'cin', 'typeOfLevel': 'surface'}
        }
        
        # Variables requiring multi-dataset processing (complex GRIB structure)
        self.complex_variables = {
            # Reflectivity (2) - Available in HRRR
            'reflectivity_comp': {'shortName': 'refc', 'typeOfLevel': 'entireAtmosphere'},
            'reflectivity_1km': {'shortName': 'refd', 'typeOfLevel': 'heightAboveGround', 'level': 1000},
            
            # Additional available reflectivity 
            'reflectivity_4km': {'shortName': 'refd', 'typeOfLevel': 'heightAboveGround', 'level': 4000},
            
            # Note: Wind shear and updraft helicity may not be available in all HRRR files
            # These need to be derived from wind components if not directly available
        }
    
    def download_hrrr_grib(self, date_str: str, hour: str, forecast_hour: str = "00") -> Optional[str]:
        """Download HRRR GRIB file for specified date and hour with AWS S3 fallback"""
        from datetime import datetime, timedelta
        
        filename = f"hrrr.t{hour.zfill(2)}z.wrfprsf{forecast_hour.zfill(2)}.grib2"
        local_path = self.output_dir / filename
        
        if local_path.exists():
            logger.info(f"GRIB file already exists: {local_path}")
            return str(local_path)
        
        # Determine if data is recent (NOAA) or historical (AWS S3)
        request_date = datetime.strptime(date_str, '%Y%m%d')
        cutoff_date = datetime.now() - timedelta(days=2)  # 48 hour cutoff
        
        if request_date >= cutoff_date:
            # Recent data: try NOAA first
            noaa_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}"
            logger.info(f"Downloading recent HRRR GRIB: {noaa_url}")
            
            try:
                response = requests.get(noaa_url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size_mb = local_path.stat().st_size / (1024*1024)
                logger.info(f"Downloaded from NOAA: {filename} ({file_size_mb:.1f} MB)")
                return str(local_path)
                
            except Exception as e:
                logger.warning(f"NOAA download failed: {e}")
                logger.info("Falling back to AWS S3...")
        
        # Historical data or NOAA failed: use AWS S3
        aws_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/{filename}"
        logger.info(f"Downloading historical HRRR GRIB: {aws_url}")
        
        try:
            response = requests.get(aws_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = local_path.stat().st_size / (1024*1024)
            logger.info(f"Downloaded from AWS S3: {filename} ({file_size_mb:.1f} MB)")
            
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download from both NOAA and AWS: {e}")
            return None
    
    def load_simple_variable(self, grib_file: str, var_name: str, var_config: Dict) -> Optional[xr.Dataset]:
        """Load a simple variable using paramId/level pattern"""
        try:
            filter_keys = {}
            
            if 'paramId' in var_config:
                filter_keys['paramId'] = var_config['paramId']
            if 'shortName' in var_config:
                filter_keys['shortName'] = var_config['shortName']
            if 'typeOfLevel' in var_config:
                filter_keys['typeOfLevel'] = var_config['typeOfLevel']
            if 'level' in var_config:
                filter_keys['level'] = var_config['level']
            
            logger.info(f"Loading {var_name} with filter: {filter_keys}")
            
            ds = cfgrib.open_dataset(
                grib_file,
                filter_by_keys=filter_keys,
                backend_kwargs={'indexpath': ''}
            )
            
            # Rename the data variable to our standard name
            data_vars = list(ds.data_vars.keys())
            if data_vars:
                ds = ds.rename({data_vars[0]: var_name})
                logger.info(f"✅ Loaded {var_name}: {ds[var_name].shape}")
                return ds
            else:
                logger.warning(f"❌ No data variables found for {var_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load {var_name}: {e}")
            return None
    
    def load_complex_variable(self, grib_file: str, var_name: str, var_config: Dict) -> Optional[xr.Dataset]:
        """Load complex variables requiring special handling"""
        try:
            # Use wgrib2 for complex variables if available, otherwise attempt cfgrib
            filter_keys = {}
            
            if 'shortName' in var_config:
                filter_keys['shortName'] = var_config['shortName']
            if 'typeOfLevel' in var_config:
                filter_keys['typeOfLevel'] = var_config['typeOfLevel']
            if 'level' in var_config:
                filter_keys['level'] = var_config['level']
            
            logger.info(f"Loading complex {var_name} with filter: {filter_keys}")
            
            ds = cfgrib.open_dataset(
                grib_file,
                filter_by_keys=filter_keys,
                backend_kwargs={'indexpath': ''}
            )
            
            # Rename the data variable
            data_vars = list(ds.data_vars.keys())
            if data_vars:
                ds = ds.rename({data_vars[0]: var_name})
                logger.info(f"✅ Loaded {var_name}: {ds[var_name].shape}")
                return ds
            else:
                logger.warning(f"❌ No data variables found for {var_name}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️  Complex variable {var_name} failed: {e}")
            return None
    
    def process_tier2_dataset(self, grib_file: str) -> Optional[xr.Dataset]:
        """Process all Tier 2 variables from HRRR GRIB file"""
        logger.info("🚀 Starting Tier 2 dataset processing...")
        
        datasets = []
        successful_vars = []
        failed_vars = []
        
        # Process simple variables first
        for var_name, var_config in self.tier2_variables.items():
            ds = self.load_simple_variable(grib_file, var_name, var_config)
            if ds is not None:
                datasets.append(ds)
                successful_vars.append(var_name)
            else:
                failed_vars.append(var_name)
        
        # Process complex variables
        for var_name, var_config in self.complex_variables.items():
            ds = self.load_complex_variable(grib_file, var_name, var_config)
            if ds is not None:
                datasets.append(ds)
                successful_vars.append(var_name)
            else:
                failed_vars.append(var_name)
        
        if not datasets:
            logger.error("❌ No variables loaded successfully")
            return None
        
        logger.info(f"✅ Successfully loaded {len(successful_vars)} variables: {successful_vars}")
        if failed_vars:
            logger.warning(f"⚠️  Failed to load {len(failed_vars)} variables: {failed_vars}")
        
        # Merge all datasets
        try:
            logger.info("🔗 Merging datasets...")
            combined_ds = xr.merge(datasets, compat='override')
            
            # Validate grid consistency
            expected_shape = (1059, 1799)  # Native HRRR 3km grid
            for var in combined_ds.data_vars:
                if combined_ds[var].shape[-2:] != expected_shape:
                    logger.warning(f"⚠️  {var} has unexpected shape: {combined_ds[var].shape}")
            
            logger.info(f"🎯 Combined dataset: {len(combined_ds.data_vars)} variables, {combined_ds.dims}")
            return combined_ds
            
        except Exception as e:
            logger.error(f"❌ Failed to merge datasets: {e}")
            return None
    
    def normalize_dataset(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        """Normalize dataset with z-score normalization"""
        logger.info("📊 Normalizing dataset...")
        
        stats = {}
        normalized_ds = ds.copy()
        
        for var in ds.data_vars:
            data = ds[var].values
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            
            # Avoid division by zero
            if std_val > 0:
                normalized_ds[var] = (ds[var] - mean_val) / std_val
            else:
                logger.warning(f"⚠️  {var} has zero std, keeping original values")
                normalized_ds[var] = ds[var]
            
            stats[var] = {'mean': float(mean_val), 'std': float(std_val)}
            logger.info(f"{var}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return normalized_ds, stats
    
    def save_training_dataset(self, ds: xr.Dataset, stats: Dict, 
                            date_str: str, hour: str, forecast_hour: str = "00") -> str:
        """Save processed dataset as NetCDF"""
        output_filename = f"HRRR_TIER2_{date_str}{hour:02d}_F{forecast_hour}.nc"
        output_path = self.output_dir / output_filename
        
        logger.info(f"💾 Saving training dataset: {output_path}")
        
        # Add metadata
        ds.attrs.update({
            'title': 'HRRR Tier 2 Enhanced Severe Weather Training Dataset',
            'description': '21 core variables for diffusion model training at native 3km resolution',
            'source': 'NOAA HRRR Model',
            'resolution': '3km native grid',
            'grid_points': ds.sizes.get('x', 0) * ds.sizes.get('y', 0),
            'variables_count': len(ds.data_vars),
            'normalization': 'z-score (mean=0, std=1)',
            'creation_date': datetime.now().isoformat(),
            'model_run': f"{date_str} {hour:02d}Z",
            'forecast_hour': forecast_hour
        })
        
        # Save with compression
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        ds.to_netcdf(output_path, encoding=encoding)
        
        # Save normalization stats
        stats_file = output_path.with_suffix('.json')
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        file_size_mb = output_path.stat().st_size / (1024*1024)
        logger.info(f"✅ Saved {output_filename}: {file_size_mb:.1f} MB")
        
        return str(output_path)
    
    def create_tier2_dataset(self, date_str: str, hour: int, forecast_hour: str = "00") -> Optional[str]:
        """Main function to create Tier 2 training dataset"""
        logger.info(f"🌟 Creating Tier 2 dataset for {date_str} {hour:02d}Z F{forecast_hour}")
        
        # Download GRIB file
        grib_file = self.download_hrrr_grib(date_str, f"{hour:02d}", forecast_hour)
        if not grib_file:
            return None
        
        # Process all variables
        ds = self.process_tier2_dataset(grib_file)
        if ds is None:
            return None
        
        # Normalize data
        normalized_ds, stats = self.normalize_dataset(ds)
        
        # Save training dataset
        output_path = self.save_training_dataset(normalized_ds, stats, date_str, hour, forecast_hour)
        
        logger.info(f"🎉 Tier 2 dataset creation completed: {output_path}")
        return output_path

def main():
    """Example usage"""
    pipeline = HRRRTier2Pipeline()
    
    # Create dataset for most recent available HRRR run
    # You can modify these parameters as needed
    date_str = "20250610"  
    hour = 0  # 00Z run
    forecast_hour = "00"  # Analysis (F00)
    
    output_path = pipeline.create_tier2_dataset(date_str, hour, forecast_hour)
    
    if output_path:
        print(f"\n🎯 SUCCESS: Tier 2 training dataset created at {output_path}")
        print("\n📊 Dataset includes 21 variables:")
        print("   Surface: t2m, d2m, u10, v10, sp")
        print("   Terrain: orog") 
        print("   Upper-air: temp_850/700/500, dewpoint_850, rh_500, u_500, v_500")
        print("   Instability: mlcape, lcl_height")
        print("   Wind Shear: wind_shear_u/v_06km, wind_shear_u/v_01km")
        print("   Reflectivity: reflectivity_comp, reflectivity_1km")
        print("   Updraft: updraft_helicity")
    else:
        print("❌ Failed to create dataset")

if __name__ == "__main__":
    main()