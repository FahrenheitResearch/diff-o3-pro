#!/usr/bin/env python
"""
Streaming HRRR downloader that enables training to start while downloading continues.
Creates a manifest file that tracks completed downloads for the data loader.
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
import json
import time
import threading

# AWS S3 configuration for anonymous access
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

class StreamingManifest:
    """Thread-safe manifest for tracking download progress."""
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.lock = threading.Lock()
        self.data = self._load()
    
    def _load(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            'completed_cycles': [],
            'partial_cycles': {},
            'total_expected_cycles': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def save(self):
        with self.lock:
            # Write to temp file first to avoid corruption
            temp_path = self.manifest_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            # Atomic rename
            temp_path.rename(self.manifest_path)
    
    def add_completed_file(self, cycle_key, forecast_hour, filepath):
        """Mark a file as completed."""
        with self.lock:
            if cycle_key not in self.data['partial_cycles']:
                self.data['partial_cycles'][cycle_key] = {
                    'forecast_hours': {},
                    'completed_count': 0
                }
            
            self.data['partial_cycles'][cycle_key]['forecast_hours'][str(forecast_hour)] = str(filepath)
            self.data['partial_cycles'][cycle_key]['completed_count'] += 1
            
            # Check if cycle is complete (has all expected hours)
            if self.data['partial_cycles'][cycle_key]['completed_count'] >= 19:  # F00-F18
                self.data['completed_cycles'].append(cycle_key)
                # Don't remove from partial - keep for reference
            
            self.save()
    
    def get_ready_cycles(self, min_hours=2):
        """Get cycles that have at least min_hours of consecutive data starting from F00."""
        with self.lock:
            ready = []
            for cycle_key, cycle_data in self.data['partial_cycles'].items():
                if cycle_key in ready:
                    continue
                
                # Check for consecutive hours starting from 0
                hours = cycle_data['forecast_hours']
                consecutive = 0
                for h in range(19):  # F00-F18
                    if str(h) in hours:
                        consecutive += 1
                    else:
                        break
                
                if consecutive >= min_hours:
                    ready.append((cycle_key, consecutive))
            
            return ready


def download_file_with_retry(s3_client, bucket, key, output_file, max_retries=3):
    """Download with retries."""
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket, key, str(output_file))
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return False
    return False


def download_forecast_hour(args):
    """Download a single forecast hour."""
    date_str, hour, forecast_hour, output_dir, manifest, cycle_key = args
    
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    bucket = 'noaa-hrrr-bdp-pds'
    key = f'hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf{forecast_hour:02d}.grib2'
    output_file = output_dir / f'hrrr.{year}{month}{day}.t{hour:02d}z.wrfprsf{forecast_hour:02d}.grib2'
    
    if output_file.exists():
        manifest.add_completed_file(cycle_key, forecast_hour, output_file)
        return True
    
    success = download_file_with_retry(s3_client, bucket, key, output_file)
    if success:
        manifest.add_completed_file(cycle_key, forecast_hour, output_file)
    else:
        # Log which files are missing
        print(f"Missing: {cycle_key} F{forecast_hour:02d}")
    
    return success


def download_streaming(start_date, end_date, hours, forecast_hours, output_dir, parallel=4):
    """Download HRRR data in streaming fashion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize manifest
    manifest_path = output_dir / 'download_manifest.json'
    manifest = StreamingManifest(manifest_path)
    
    # Generate all download tasks
    tasks = []
    current_date = start_date
    total_cycles = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        for hour in hours:
            cycle_key = f"{date_str}_{hour:02d}"
            total_cycles += 1
            
            for fh in forecast_hours:
                tasks.append((date_str, hour, fh, output_dir, manifest, cycle_key))
        
        current_date += timedelta(days=1)
    
    manifest.data['total_expected_cycles'] = total_cycles
    manifest.save()
    
    print(f"Starting streaming download of {len(tasks)} files ({total_cycles} cycles)")
    print(f"Training can begin once a cycle has 2+ consecutive hours")
    print(f"Manifest: {manifest_path}")
    
    # Start downloading with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        # Submit all tasks
        futures = [executor.submit(download_forecast_hour, task) for task in tasks]
        
        # Progress bar
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    pbar.update(1)
                    
                    # Periodically report ready cycles
                    if pbar.n % 10 == 0:
                        ready = manifest.get_ready_cycles(min_hours=2)
                        if ready:
                            total_hours = sum(hours for _, hours in ready)
                            pbar.set_postfix({'ready_cycles': len(ready), 'total_hours': total_hours})
                except Exception as e:
                    pbar.update(1)
                    print(f"\nError in download: {e}")
    
    print("\nDownload complete!")
    ready = manifest.get_ready_cycles(min_hours=2)
    print(f"Ready cycles: {len(ready)} with {sum(h for _, h in ready)} total hours")


def main():
    parser = argparse.ArgumentParser(description='Streaming HRRR downloader')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYYMMDD), defaults to start date')
    parser.add_argument('--hours', nargs='+', type=int, default=[0, 6, 12, 18], help='UTC hours to download')
    parser.add_argument('--forecast-hours', type=str, default='0-18', help='Forecast hours to download')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/streaming'), help='Output directory')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d') if args.end_date else start_date
    
    # Parse forecast hours
    if '-' in args.forecast_hours:
        start_fh, end_fh = map(int, args.forecast_hours.split('-'))
        forecast_hours = list(range(start_fh, end_fh + 1))
    else:
        forecast_hours = [int(args.forecast_hours)]
    
    # Start streaming download
    download_streaming(start_date, end_date, args.hours, forecast_hours, args.output_dir, args.parallel)


if __name__ == '__main__':
    main()