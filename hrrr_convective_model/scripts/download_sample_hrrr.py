#!/usr/bin/env python
"""
Download a few sample HRRR files to test the pipeline.
This downloads REAL data from the last 24 hours.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Configure anonymous S3 access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_recent_hrrr(output_dir, num_files=4):
    """Download the most recent available HRRR files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to get files from the last 48 hours
    now = datetime.utcnow()
    downloaded = 0
    
    for hours_ago in range(0, 48, 6):  # Try every 6 hours
        if downloaded >= num_files:
            break
            
        file_time = now - timedelta(hours=hours_ago)
        date_str = file_time.strftime('%Y%m%d')
        hour_str = file_time.strftime('%H')
        
        # Round to nearest 6-hour cycle
        hour = int(hour_str)
        hour = (hour // 6) * 6
        
        # AWS S3 path
        bucket = 'noaa-hrrr-bdp-pds'
        key = f'hrrr.{date_str}/conus/hrrr.t{hour:02d}z.wrfprsf00.grib2'
        output_file = output_dir / f'hrrr.{date_str}.t{hour:02d}z.wrfprsf00.grib2'
        
        if output_file.exists():
            print(f"Already exists: {output_file.name}")
            downloaded += 1
            continue
        
        try:
            print(f"Downloading {key}...")
            s3.download_file(bucket, key, str(output_file))
            print(f"✓ Downloaded {output_file.name}")
            downloaded += 1
        except Exception as e:
            print(f"✗ Failed to download {key}: {e}")
            # Try alternate source
            try:
                url = f"https://storage.googleapis.com/high-resolution-rapid-refresh/{key}"
                print(f"  Trying Google Cloud: {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    print(f"  ✓ Downloaded from GCS")
                    downloaded += 1
            except Exception as e2:
                print(f"  ✗ Also failed from GCS: {e2}")
    
    if downloaded == 0:
        print("\nERROR: Could not download any HRRR files!")
        print("Please check your internet connection.")
        sys.exit(1)
    
    print(f"\nSuccessfully downloaded {downloaded} HRRR files to {output_dir}")
    return downloaded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--num-files", type=int, default=4, help="Number of files to download")
    args = parser.parse_args()
    
    download_recent_hrrr(args.output_dir, args.num_files)