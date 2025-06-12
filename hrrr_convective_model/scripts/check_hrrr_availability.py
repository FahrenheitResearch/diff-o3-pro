#!/usr/bin/env python
"""
Check which HRRR forecast hours are available for a given date/cycle.
This helps determine what data is actually available on AWS S3.
"""
import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime

# AWS S3 configuration for anonymous access
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def check_forecast_availability(date_str, hour):
    """Check which forecast hours are available for a given cycle."""
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    bucket = 'noaa-hrrr-bdp-pds'
    prefix = f'hrrr.{year}{month}{day}/conus/hrrr.t{hour:02d}z.wrfprsf'
    
    available = []
    missing = []
    
    print(f"\nChecking {date_str} {hour:02d}Z cycle:")
    print("-" * 40)
    
    # Check each forecast hour
    for fh in range(49):  # Check up to F48
        key = f'{prefix}{fh:02d}.grib2'
        try:
            # Just check if object exists
            s3_client.head_object(Bucket=bucket, Key=key)
            available.append(fh)
            print(f"F{fh:02d}: ✓ Available")
        except:
            missing.append(fh)
            if fh <= 18:  # Only print missing for F00-F18
                print(f"F{fh:02d}: ✗ Missing")
    
    print(f"\nSummary:")
    print(f"Available: F00-F{available[-1]:02d} ({len(available)} hours)")
    if missing and missing[0] <= 18:
        print(f"Missing in F00-F18 range: {[f'F{h:02d}' for h in missing if h <= 18]}")
    
    return available, missing

def main():
    parser = argparse.ArgumentParser(description='Check HRRR forecast availability')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYYMMDD)')
    parser.add_argument('--hour', type=int, default=0, help='Cycle hour (0-23)')
    parser.add_argument('--check-all-cycles', action='store_true', help='Check all 4 main cycles')
    
    args = parser.parse_args()
    
    if args.check_all_cycles:
        cycles = [0, 6, 12, 18]
        all_results = {}
        
        for hour in cycles:
            available, missing = check_forecast_availability(args.date, hour)
            all_results[hour] = {
                'available': len(available),
                'max_forecast': available[-1] if available else -1
            }
        
        print("\n" + "="*50)
        print("SUMMARY FOR ALL CYCLES:")
        print("="*50)
        for hour, result in all_results.items():
            print(f"{hour:02d}Z: {result['available']} hours available (F00-F{result['max_forecast']:02d})")
    else:
        check_forecast_availability(args.date, args.hour)

if __name__ == '__main__':
    main()