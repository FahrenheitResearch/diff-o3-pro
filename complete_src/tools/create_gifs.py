#!/usr/bin/env python3
"""
HRRR GIF Creation Utility
Automatically create GIFs for all processed parameters in a model run
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def find_all_parameters(output_base, date, hour):
    """
    Find all parameters that were processed for a given model run.
    
    Args:
        output_base: Base output directory
        date: Date string (YYYYMMDD)
        hour: Hour string (XXz)
        
    Returns:
        Dictionary of {category: [parameter_list]}
    """
    model_dir = Path(output_base) / 'outputs' / 'hrrr' / date / hour
    
    if not model_dir.exists():
        return {}
    
    parameters = {}
    
    # Look for parameters in nested structure (F00/F00/category/)
    nested_f00_dir = model_dir / 'F00' / 'F00'
    if nested_f00_dir.exists():
        for category_dir in nested_f00_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                params = []

                for png_file in category_dir.glob('*_f00_REFACTORED.png'):
                    filename = png_file.name
                    param_name = filename.replace('_f00_REFACTORED.png', '')
                    params.append(param_name)

                if params:
                    parameters.setdefault(category, set()).update(params)
    
    # Look for parameters in flat category structure (F00/category/)
    flat_f00_dir = model_dir / 'F00'
    if flat_f00_dir.exists():
        for item in flat_f00_dir.iterdir():
            if item.is_dir() and item.name != 'F00':  # Skip nested F00 dir
                category = item.name
                params = []

                for png_file in item.glob('*_f00_REFACTORED.png'):
                    filename = png_file.name
                    param_name = filename.replace('_f00_REFACTORED.png', '')
                    params.append(param_name)

                if params:
                    parameters.setdefault(category, set()).update(params)
    
    # Look for parameters directly in forecast hour directory (flat structure)
    if flat_f00_dir.exists():
        params = []
        for png_file in flat_f00_dir.glob('*_f00_REFACTORED.png'):
            filename = png_file.name
            param_name = filename.replace('_f00_REFACTORED.png', '')
            params.append(param_name)
        
        if params:
            # Group all flat files under a generic category
            parameters.setdefault('all', set()).update(params)

    # Convert sets to sorted lists
    parameters = {k: sorted(v) for k, v in parameters.items()}
    
    return parameters

def create_gifs_for_model_run(output_base, date, hour, max_hours=48, duration=500, categories=None):
    """
    Create GIFs for all parameters in a model run.
    
    Args:
        output_base: Base output directory
        date: Date string (YYYYMMDD)
        hour: Hour string (XXz) 
        max_hours: Maximum forecast hours to include
        duration: Frame duration in milliseconds
        categories: List of categories to process (None = all)
    """
    print(f"🎬 Creating GIFs for {date} {hour} model run")
    
    # Find all parameters
    parameters = find_all_parameters(output_base, date, hour)
    
    if not parameters:
        print(f"❌ No parameters found for {date} {hour}")
        return
    
    # Filter categories if specified
    if categories:
        parameters = {k: v for k, v in parameters.items() if k in categories}
    
    print(f"📊 Found parameters in {len(parameters)} categories:")
    total_params = 0
    for category, params in parameters.items():
        print(f"   {category}: {len(params)} parameters")
        total_params += len(params)
    
    print(f"🎯 Total: {total_params} GIFs to create")
    
    # Create output directory for GIFs
    gif_dir = Path(output_base) / 'outputs' / 'hrrr' / date / hour / 'animations'
    gif_dir.mkdir(exist_ok=True)
    
    created_count = 0
    failed_count = 0
    
    # Process each parameter
    for category, params in parameters.items():
        print(f"\n📁 Processing {category} category...")
        
        category_gif_dir = gif_dir / category
        category_gif_dir.mkdir(exist_ok=True)
        
        for param in params:
            try:
                # Find a sample file to pass to the GIF maker
                sample_file_nested = Path(output_base) / 'outputs' / 'hrrr' / date / hour / 'F00' / 'F00' / category / f'{param}_f00_REFACTORED.png'
                sample_file_flat = Path(output_base) / 'outputs' / 'hrrr' / date / hour / 'F00' / category / f'{param}_f00_REFACTORED.png'
                sample_file_direct = Path(output_base) / 'outputs' / 'hrrr' / date / hour / 'F00' / f'{param}_f00_REFACTORED.png'

                if sample_file_nested.exists():
                    sample_file = sample_file_nested
                elif sample_file_flat.exists():
                    sample_file = sample_file_flat
                elif sample_file_direct.exists():
                    sample_file = sample_file_direct
                else:
                    print(f"  ⚠️  Sample file not found: {param}")
                    failed_count += 1
                    continue
                
                # Generate output filename
                output_gif = category_gif_dir / f'{param}_{date}_{hour}_animation.gif'
                
                # Skip if GIF already exists
                if output_gif.exists():
                    print(f"  ⏭️  Skipping {param} (GIF exists)")
                    continue
                
                print(f"  🎬 Creating {param}...")
                
                # Call the GIF maker script
                cmd = [
                    sys.executable, 'tools/hrrr_gif_maker.py',
                    str(sample_file),
                    '--max-hours', str(max_hours),
                    '--duration', str(duration),
                    '--output', str(output_gif),
                    '--base-dir', output_base
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
                
                if result.returncode == 0:
                    # Get file size
                    if output_gif.exists():
                        size_mb = output_gif.stat().st_size / (1024 * 1024)
                        print(f"     ✅ Success ({size_mb:.1f} MB)")
                        created_count += 1
                    else:
                        print(f"     ❌ Failed (no output file)")
                        failed_count += 1
                else:
                    print(f"     ❌ Failed: {result.stderr.strip()}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
                failed_count += 1
    
    print(f"\n🎉 GIF creation complete!")
    print(f"   ✅ Created: {created_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Output: {gif_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIFs for all parameters in an HRRR model run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create GIFs for all parameters in a model run
  python create_gifs.py 20250604 18z
  
  # Create GIFs for specific categories only
  python create_gifs.py 20250604 18z --categories personality,severe
  
  # Create shorter animations with faster frame rate
  python create_gifs.py 20250604 18z --max-hours 12 --duration 300
        """
    )
    
    parser.add_argument('date', help='Model date (YYYYMMDD)')
    parser.add_argument('hour', help='Model hour (e.g., 18z)')
    
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory containing outputs/ (default: current)'
    )
    
    parser.add_argument(
        '--categories',
        help='Comma-separated list of categories to process (default: all)'
    )
    
    parser.add_argument(
        '--max-hours',
        type=int,
        default=48,
        help='Maximum forecast hours to include (default: 48)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=500,
        help='Frame duration in milliseconds (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
    
    # Validate date and hour
    try:
        datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        return 1
    
    if not args.hour.endswith('z') or not args.hour[:-1].isdigit():
        print(f"❌ Invalid hour format: {args.hour}")
        return 1
    
    # Create GIFs
    try:
        create_gifs_for_model_run(
            args.base_dir, 
            args.date, 
            args.hour, 
            args.max_hours,
            args.duration,
            categories
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())