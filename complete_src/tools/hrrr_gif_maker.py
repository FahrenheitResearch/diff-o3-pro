#!/usr/bin/env python3
"""
HRRR GIF Maker - Automated Weather Animation Generator

Creates animated GIFs from HRRR weather parameter forecast sequences.
Automatically detects available forecast hours and generates smooth animations.

Usage:
    python hrrr_gif_maker.py "outputs/hrrr/20250604/18z/F00/F00/personality/destroyer_reality_check_f00_REFACTORED.png"
    python hrrr_gif_maker.py "destroyer_reality_check_f00_REFACTORED.png" --date 20250604 --hour 18z
    python hrrr_gif_maker.py --interactive
"""

import os
import re
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def parse_path_components(file_path: str) -> Dict[str, str]:
    """
    Parse file path to extract date, hour, and parameter information.
    
    Args:
        file_path: Full or relative path to PNG file
        
    Returns:
        Dictionary with parsed components
    """
    # Convert to Path object for easier handling
    path = Path(file_path)
    
    # Extract filename and components
    filename = path.name
    
    # Parse filename pattern: parameter_fXX_REFACTORED.png
    filename_match = re.match(r'(.+)_f(\d+)_REFACTORED\.png', filename)
    if not filename_match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    
    parameter_name = filename_match.group(1)
    forecast_hour = filename_match.group(2)
    
    # Try to extract date and hour from path
    path_str = str(path)
    
    # Look for date pattern (YYYYMMDD)
    date_match = re.search(r'(\d{8})', path_str)
    date = date_match.group(1) if date_match else None
    
    # Look for hour pattern (XXz)
    hour_match = re.search(r'(\d{2}z)', path_str)
    hour = hour_match.group(1) if hour_match else None
    
    # Extract category from path if present
    category = None
    if '/personality/' in path_str:
        category = 'personality'
    elif '/severe/' in path_str:
        category = 'severe'
    elif '/surface/' in path_str:
        category = 'surface'
    elif '/reflectivity/' in path_str:
        category = 'reflectivity'
    elif '/smoke/' in path_str:
        category = 'smoke'
    elif '/instability/' in path_str:
        category = 'instability'
    elif '/atmospheric/' in path_str:
        category = 'atmospheric'
    elif '/precipitation/' in path_str:
        category = 'precipitation'
    elif '/derived/' in path_str:
        category = 'derived'
    elif '/advanced/' in path_str:
        category = 'advanced'
    elif '/custom/' in path_str:
        category = 'custom'
    
    return {
        'parameter_name': parameter_name,
        'forecast_hour': forecast_hour,
        'date': date,
        'hour': hour,
        'category': category,
        'filename': filename,
        'original_path': str(path)
    }

def find_forecast_sequence(base_path: str, parameter_name: str, category: str, 
                          date: str, hour: str) -> List[str]:
    """
    Find all available forecast hours for the given parameter.
    
    Args:
        base_path: Base output directory path
        parameter_name: Name of the weather parameter
        category: Parameter category (personality, severe, etc.)
        date: Date string (YYYYMMDD)
        hour: Hour string (XXz)
        
    Returns:
        List of file paths sorted by forecast hour
    """
    # Construct search patterns for different directory structures
    patterns = []
    
    if category and category != 'all':
        # Search in category subdirectories
        patterns.extend([
            os.path.join(
                base_path, 'outputs', 'hrrr', date, hour, 'F*', 'F*', category,
                f'{parameter_name}_f*_REFACTORED.png'
            ),
            os.path.join(
                base_path, 'outputs', 'hrrr', date, hour, 'F*', category,
                f'{parameter_name}_f*_REFACTORED.png'
            ),
        ])
    
    # Always search directly in forecast hour directories (flat structure)
    patterns.append(
        os.path.join(
            base_path, 'outputs', 'hrrr', date, hour, 'F*',
            f'{parameter_name}_f*_REFACTORED.png'
        )
    )
    
    # If no category specified, search in all subdirectories
    if not category:
        patterns.extend([
            os.path.join(
                base_path, 'outputs', 'hrrr', date, hour, 'F*', 'F*', '*',
                f'{parameter_name}_f*_REFACTORED.png'
            ),
            os.path.join(
                base_path, 'outputs', 'hrrr', date, hour, 'F*', '*',
                f'{parameter_name}_f*_REFACTORED.png'
            ),
        ])

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    # Sort by forecast hour
    def extract_forecast_hour(filepath):
        match = re.search(r'_f(\d+)_REFACTORED\.png', filepath)
        return int(match.group(1)) if match else 0
    
    unique_files.sort(key=extract_forecast_hour)
    return unique_files

def create_gif(image_files: List[str], output_path: str, duration: int = 500,
               loop: int = 0) -> None:
    """
    Create animated GIF from sequence of PNG files.
    
    Args:
        image_files: List of PNG file paths in sequence order
        output_path: Output GIF file path
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for GIF creation. Install with: pip install Pillow")
    
    if not image_files:
        raise ValueError("No image files provided")
    
    print(f"Creating GIF with {len(image_files)} frames...")
    
    # Load all images
    images = []
    for i, filepath in enumerate(image_files):
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
            
        try:
            img = Image.open(filepath)
            images.append(img)
            print(f"  Loaded frame {i+1}/{len(image_files)}: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    # Create GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"✅ GIF created: {output_path}")
    print(f"   Frames: {len(images)}")
    print(f"   Duration: {duration}ms per frame")
    print(f"   Total time: {len(images) * duration / 1000:.1f} seconds")

def interactive_mode():
    """Interactive mode for GIF creation."""
    print("\n🎬 HRRR GIF Maker - Interactive Mode")
    print("=" * 50)
    
    # Get base directory
    base_dir = input("Enter base directory (default: current): ").strip()
    if not base_dir:
        base_dir = "."
    
    # Get date
    date = input("Enter date (YYYYMMDD): ").strip()
    if not re.match(r'\d{8}', date):
        print("❌ Invalid date format")
        return
    
    # Get hour
    hour = input("Enter model hour (e.g., 18z): ").strip()
    if not re.match(r'\d{2}z', hour):
        print("❌ Invalid hour format")
        return
    
    # List available categories
    print("\n📁 Scanning for available categories...")
    categories_found = set()
    
    model_dir = os.path.join(base_dir, 'outputs', 'hrrr', date, hour)
    if os.path.exists(model_dir):
        for forecast_dir in os.listdir(model_dir):
            forecast_path = os.path.join(model_dir, forecast_dir)
            if os.path.isdir(forecast_path) and forecast_dir.startswith('F'):
                for subdir in os.listdir(forecast_path):
                    subpath = os.path.join(forecast_path, subdir)
                    if os.path.isdir(subpath):
                        if subdir.startswith('F'):
                            for category in os.listdir(subpath):
                                cat_path = os.path.join(subpath, category)
                                if os.path.isdir(cat_path):
                                    categories_found.add(category)
                        else:
                            categories_found.add(subdir)
    
    if categories_found:
        print("Available categories:")
        for i, cat in enumerate(sorted(categories_found), 1):
            print(f"  {i}. {cat}")
        
        # Get category choice
        try:
            choice = int(input(f"\nSelect category (1-{len(categories_found)}): "))
            category = sorted(categories_found)[choice - 1]
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return
    else:
        print("❌ No categories found")
        return
    
    # List available parameters in category
    print(f"\n📊 Scanning for parameters in '{category}' category...")
    parameters_found = set()
    
    sample_dir_nested = os.path.join(model_dir, 'F00', 'F00', category)
    sample_dir_flat = os.path.join(model_dir, 'F00', category)
    for sdir in [sample_dir_nested, sample_dir_flat]:
        if os.path.exists(sdir):
            for file in os.listdir(sdir):
                if file.endswith('_REFACTORED.png'):
                    match = re.match(r'(.+)_f\d+_REFACTORED\.png', file)
                    if match:
                        parameters_found.add(match.group(1))
    
    if parameters_found:
        print("Available parameters:")
        for i, param in enumerate(sorted(parameters_found), 1):
            print(f"  {i}. {param}")
        
        # Get parameter choice
        try:
            choice = int(input(f"\nSelect parameter (1-{len(parameters_found)}): "))
            parameter = sorted(parameters_found)[choice - 1]
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return
    else:
        print("❌ No parameters found")
        return
    
    # Get animation settings
    print("\n⚙️ Animation Settings")
    try:
        duration = int(input("Frame duration in ms (default: 500): ") or "500")
        max_hours = int(input("Maximum forecast hours (default: 48): ") or "48")
    except ValueError:
        print("❌ Invalid settings")
        return
    
    # Find files and create GIF
    files = find_forecast_sequence(base_dir, parameter, category, date, hour)
    
    # Filter by max hours
    files = [f for f in files if extract_forecast_hour_from_path(f) <= max_hours]
    
    if not files:
        print("❌ No files found matching criteria")
        return
    
    print(f"\n🎞️ Found {len(files)} frames")
    
    # Generate output filename
    output_filename = f"{parameter}_{date}_{hour}_animation.gif"
    output_path = os.path.join(base_dir, output_filename)
    
    # Create GIF
    try:
        create_gif(files, output_path, duration)
        print(f"\n🎉 Success! GIF saved as: {output_filename}")
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")

def extract_forecast_hour_from_path(filepath: str) -> int:
    """Extract forecast hour from file path."""
    match = re.search(r'_f(\d+)_REFACTORED\.png', filepath)
    return int(match.group(1)) if match else 0

def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIFs from HRRR weather parameter sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create GIF from specific file (auto-detects sequence)
  python hrrr_gif_maker.py "outputs/hrrr/20250604/18z/F00/F00/personality/destroyer_reality_check_f00_REFACTORED.png"
  
  # Create GIF with custom settings
  python hrrr_gif_maker.py "destroyer_reality_check_f00_REFACTORED.png" --date 20250604 --hour 18z --duration 300 --max-hours 24
  
  # Interactive mode
  python hrrr_gif_maker.py --interactive
        """
    )
    
    # Positional argument for file path (optional in interactive mode)
    parser.add_argument(
        'file_path',
        nargs='?',
        help='Path to sample PNG file for parameter detection'
    )
    
    # Optional arguments
    parser.add_argument(
        '--date',
        help='Model date (YYYYMMDD) - overrides date from path'
    )
    
    parser.add_argument(
        '--hour',
        help='Model hour (e.g., 18z) - overrides hour from path'
    )
    
    parser.add_argument(
        '--category',
        help='Parameter category - overrides category from path'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=500,
        help='Frame duration in milliseconds (default: 500)'
    )
    
    parser.add_argument(
        '--max-hours',
        type=int,
        default=48,
        help='Maximum forecast hours to include (default: 48)'
    )
    
    parser.add_argument(
        '--output',
        help='Output GIF filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory for file search (default: current directory)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List available files without creating GIF'
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Require file path for non-interactive mode
    if not args.file_path:
        parser.error("file_path is required unless using --interactive mode")
    
    try:
        # Parse components from provided path
        components = parse_path_components(args.file_path)
        
        # Override with command line arguments if provided
        date = args.date or components['date']
        hour = args.hour or components['hour']
        category = args.category or components['category']
        parameter_name = components['parameter_name']
        
        if not date:
            print("❌ Could not determine date. Please specify with --date")
            return
        
        if not hour:
            print("❌ Could not determine hour. Please specify with --hour")
            return
        
        print(f"🔍 Searching for forecast sequence:")
        print(f"   Parameter: {parameter_name}")
        print(f"   Date: {date}")
        print(f"   Hour: {hour}")
        print(f"   Category: {category or 'auto-detect'}")
        print(f"   Max hours: {args.max_hours}")
        
        # Find all files in sequence
        files = find_forecast_sequence(args.base_dir, parameter_name, category, date, hour)
        
        # Filter by max hours
        files = [f for f in files if extract_forecast_hour_from_path(f) <= args.max_hours]
        
        if not files:
            print("❌ No files found matching the specified criteria")
            print("\nTroubleshooting:")
            print("1. Check that the base directory is correct")
            print("2. Verify the date and hour exist in outputs/")
            print("3. Ensure the parameter name is exact")
            print("4. Try using --interactive mode to browse available options")
            return
        
        print(f"\n📁 Found {len(files)} files:")
        for i, f in enumerate(files[:5]):  # Show first 5
            forecast_hour = extract_forecast_hour_from_path(f)
            print(f"   F{forecast_hour:02d}: {os.path.basename(f)}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
        
        # List only mode
        if args.list_only:
            print(f"\n📋 Complete file list:")
            for f in files:
                forecast_hour = extract_forecast_hour_from_path(f)
                print(f"   F{forecast_hour:02d}: {f}")
            return
        
        # Generate output filename
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{parameter_name}_{date}_{hour}_{timestamp}.gif"
            output_path = os.path.join(args.base_dir, output_filename)
        
        # Create GIF
        print(f"\n🎬 Creating animated GIF...")
        create_gif(files, output_path, args.duration)
        
        print(f"\n🎉 Animation complete!")
        print(f"   Output: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())