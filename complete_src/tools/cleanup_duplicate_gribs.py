#!/usr/bin/env python3
"""
Clean up duplicate GRIB files and organize them centrally
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def find_duplicate_gribs():
    """Find all GRIB files and group by name"""
    grib_files = defaultdict(list)
    
    # Search current directory for GRIB files
    for grib_file in Path(".").glob("**/*.grib2"):
        if grib_file.is_file():
            filename = grib_file.name
            file_size = grib_file.stat().st_size
            grib_files[filename].append((grib_file, file_size))
    
    return grib_files

def cleanup_duplicates():
    """Clean up duplicate GRIB files"""
    print("🧹 Cleaning up duplicate GRIB files...")
    
    grib_files = find_duplicate_gribs()
    
    total_size_mb = 0
    duplicates_found = 0
    
    for filename, file_list in grib_files.items():
        if len(file_list) > 1:
            print(f"\n📁 Found {len(file_list)} copies of {filename}:")
            
            # Sort by file size (largest first) to keep the complete file
            file_list.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the first (largest) file, delete the rest
            keep_file, keep_size = file_list[0]
            print(f"  ✅ Keeping: {keep_file} ({keep_size/1024/1024:.1f}MB)")
            
            for duplicate_file, dup_size in file_list[1:]:
                print(f"  🗑️  Deleting: {duplicate_file} ({dup_size/1024/1024:.1f}MB)")
                try:
                    duplicate_file.unlink()
                    total_size_mb += dup_size / 1024 / 1024
                    duplicates_found += 1
                except Exception as e:
                    print(f"    ❌ Error deleting {duplicate_file}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"  Duplicates removed: {duplicates_found}")
    print(f"  Space freed: {total_size_mb:.1f}MB")

def organize_gribs_centrally():
    """Organize remaining GRIB files into central directory"""
    print(f"\n📁 Organizing GRIB files centrally...")
    
    grib_files = find_duplicate_gribs()
    moved_count = 0
    
    for filename, file_list in grib_files.items():
        if file_list:
            source_file, file_size = file_list[0]  # Take the first (and now only) file
            
            # Extract cycle info from filename
            # Format: hrrr.t20z.wrfprsf01.grib2
            try:
                parts = filename.split('.')
                if len(parts) >= 4 and parts[0] == 'hrrr':
                    cycle_part = parts[1]  # t20z
                    file_type_part = parts[2]  # wrfprsf01
                    
                    # Extract hour and forecast hour
                    hour = cycle_part[1:3]  # 20
                    fhr = file_type_part[-2:]  # 01
                    
                    # Assume today's date for organization
                    from datetime import datetime
                    today = datetime.now().strftime('%Y%m%d')
                    
                    # Create organized directory
                    target_dir = Path("grib_files") / today / f"{hour}z"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_file = target_dir / filename
                    
                    # Move file if it's not already in the right place
                    if source_file.resolve() != target_file.resolve():
                        print(f"  📦 Moving {filename} -> {target_dir}")
                        shutil.move(str(source_file), str(target_file))
                        moved_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ Could not organize {filename}: {e}")
    
    print(f"📊 Organization Summary:")
    print(f"  Files organized: {moved_count}")

if __name__ == '__main__':
    cleanup_duplicates()
    organize_gribs_centrally()
    print(f"\n✅ Cleanup complete!")