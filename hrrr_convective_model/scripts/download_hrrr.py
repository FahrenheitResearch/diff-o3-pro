#!/usr/bin/env python
"""
Download REAL HRRR GRIB2 files - NO SAMPLE DATA
Supports multiple sources: AWS, Google Cloud, NCEP
"""
import os
import sys
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import concurrent.futures
from tqdm import tqdm

# AWS S3 configuration for anonymous access
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def download_from_aws(date_str, hour, output_dir, forecast_hours=None):
    """Download HRRR from AWS S3
    
    Args:
        date_str: Date in YYYYMMDD format
        hour: Model initialization hour (0-23)
        output_dir: Output directory
        forecast_hours: List of forecast hours to download (default: [0])
    """
    if forecast_hours is None:
        forecast_hours = [0]
        
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    bucket = 'noaa-hrrr-bdp-pds'
    
    success_count = 0
    for fhr in forecast_hours:
        key = f'hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        output_file = output_dir / f'hrrr.{year}{month}{day}.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        
        if output_file.exists():
            print(f"File already exists: {output_file}")
            success_count += 1
            continue
        
        try:
            print(f"Downloading from AWS S3: {bucket}/{key}")
            s3_client.download_file(bucket, key, str(output_file))
            success_count += 1
        except Exception as e:
            print(f"AWS download failed for F{fhr:02d}: {e}")
    
    return success_count == len(forecast_hours)

def download_from_gcs(date_str, hour, output_dir, forecast_hours=None):
    """Download HRRR from Google Cloud Storage"""
    if forecast_hours is None:
        forecast_hours = [0]
        
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    success_count = 0
    
    for fhr in forecast_hours:
        url = f'https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        output_file = output_dir / f'hrrr.{year}{month}{day}.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        
        if output_file.exists():
            print(f"File already exists: {output_file}")
            success_count += 1
            continue
        
        try:
            print(f"Downloading from GCS: {url}")
            download_file(url, output_file)
            success_count += 1
        except Exception as e:
            print(f"GCS download failed for F{fhr:02d}: {e}")
    
    return success_count == len(forecast_hours)

def download_from_ncep(date_str, hour, output_dir):
    """Download HRRR from NCEP"""
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    
    # Try current operational data first
    url = f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf00.grib2'
    output_file = output_dir / f'hrrr.{year}{month}{day}.t{hour:02d}z.wrfprsf00.grib2'
    
    if output_file.exists():
        print(f"File already exists: {output_file}")
        return True
    
    try:
        print(f"Downloading from NCEP: {url}")
        download_file(url, output_file)
        return True
    except Exception as e:
        print(f"NCEP download failed: {e}")
        return False

def download_forecast_files(date_str, hour, output_dir, max_forecast=18):
    """Download forecast files f01-f18"""
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    
    for fhr in range(1, max_forecast + 1):
        # Try AWS first
        bucket = 'noaa-hrrr-bdp-pds'
        key = f'hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        output_file = output_dir / f'hrrr.{year}{month}{day}.t{hour:02d}z.wrfprsf{fhr:02d}.grib2'
        
        if output_file.exists():
            continue
            
        try:
            print(f"Downloading forecast f{fhr:02d} from AWS")
            s3_client.download_file(bucket, key, str(output_file))
        except Exception as e:
            print(f"Failed to download forecast f{fhr:02d}: {e}")

def download_hrrr_file(args):
    """Download a single HRRR file trying multiple sources"""
    date_str, hour, output_dir, forecast_hours = args
    
    # Try sources in order
    if download_from_aws(date_str, hour, output_dir, forecast_hours):
        return True
    if download_from_gcs(date_str, hour, output_dir, forecast_hours):
        return True
    # NCEP usually only has F00
    if len(forecast_hours) == 1 and forecast_hours[0] == 0:
        if download_from_ncep(date_str, hour, output_dir):
            return True
    
    print(f"Failed to download HRRR for {date_str} {hour:02d}Z from all sources")
    return False

def main():
    parser = argparse.ArgumentParser(description='Download REAL HRRR GRIB2 data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYYMMDD), defaults to start date')
    parser.add_argument('--hours', nargs='+', type=int, default=[0, 6, 12, 18], help='UTC hours to download')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw'), help='Output directory')
    parser.add_argument('--forecast-hours', type=str, default='0', help='Forecast hours to download (e.g., "0-18" or "0,6,12,18")')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d') if args.end_date else start_date
    
    # Parse forecast hours
    if '-' in args.forecast_hours:
        # Range like "0-18"
        start_fh, end_fh = map(int, args.forecast_hours.split('-'))
        forecast_hours = list(range(start_fh, end_fh + 1))
    elif ',' in args.forecast_hours:
        # List like "0,6,12,18"
        forecast_hours = list(map(int, args.forecast_hours.split(',')))
    else:
        # Single hour
        forecast_hours = [int(args.forecast_hours)]
    
    print(f"Will download forecast hours: {forecast_hours}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate download tasks
    tasks = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        for hour in args.hours:
            tasks.append((date_str, hour, args.output_dir, forecast_hours))
        current_date += timedelta(days=1)
    
    print(f"Downloading {len(tasks)} HRRR cycles x {len(forecast_hours)} forecast hours...")
    
    # Download in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        list(tqdm(executor.map(download_hrrr_file, tasks), total=len(tasks)))
    
    print(f"Download complete. Files saved to {args.output_dir}")

if __name__ == '__main__':
    main()