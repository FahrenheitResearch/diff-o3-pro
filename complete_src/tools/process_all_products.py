#!/usr/bin/env python3
"""
Process All HRRR Products
Generates maps for all available HRRR parameters
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path so we can import from the main project
sys.path.insert(0, str(Path(__file__).parent.parent))
from hrrr_processor_refactored import HRRRProcessor


def process_all_products(cycle=None, forecast_hours=[1], output_dir=None, 
                        by_category=False, include_categories=None, 
                        exclude_categories=None, grib_dir=None, use_local_grib=False):
    """
    Process all available HRRR products
    
    Args:
        cycle: HRRR cycle (YYYYMMDDHH) or None for recent
        forecast_hours: List of forecast hours to process
        output_dir: Output directory
        by_category: Process one category at a time for organization
        include_categories: Only process these categories
        exclude_categories: Skip these categories
    """
    
    processor = HRRRProcessor()
    
    # Set GRIB directory if provided
    if grib_dir:
        processor.grib_dir = grib_dir
    
    # Set use_local_grib flag
    if use_local_grib:
        processor.use_local_grib = True
    
    # Setup output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'./all_products_{timestamp}')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all available categories and fields
    all_categories = processor.registry.get_available_categories()
    all_fields = processor.registry.get_all_fields()
    
    # Filter categories if specified
    if include_categories:
        categories_to_process = [cat for cat in all_categories if cat in include_categories]
    else:
        categories_to_process = all_categories
        
    if exclude_categories:
        categories_to_process = [cat for cat in categories_to_process if cat not in exclude_categories]
    
    print(f"🚀 PROCESSING ALL HRRR PRODUCTS")
    print(f"=" * 60)
    print(f"Categories to process: {categories_to_process}")
    print(f"Forecast hours: {forecast_hours}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)
    
    total_success = 0
    total_attempted = 0
    
    # Process each forecast hour
    for fhr in forecast_hours:
        print(f"\n📅 Processing forecast hour F{fhr:02d}")
        
        # Create subdirectory for this forecast hour
        fhr_output_dir = output_dir / f"F{fhr:02d}"
        fhr_output_dir.mkdir(exist_ok=True)
        
        if by_category:
            # Process by category for better organization
            import time
            last_category_end = time.time()
            
            for category in categories_to_process:
                # Add detailed category timing
                current_time = time.time()
                if last_category_end > 0:
                    gap = current_time - last_category_end
                    print(f"🕐 Gap before {category}: {gap:.3f}s")
                    if gap > 10.0:
                        print(f"⏱️  CATEGORY GAP: {gap:.1f}s before {category}")
                
                print(f"\n📂 Processing category: {category.upper()} at {time.strftime('%H:%M:%S')}")
                category_start = time.time()
                
                # Create category subdirectory
                cat_output_dir = fhr_output_dir / category
                cat_output_dir.mkdir(exist_ok=True)
                
                # Get fields in this category
                category_fields = processor.registry.get_fields_by_category(category)
                
                if category_fields:
                    print(f"🔄 Starting {len(category_fields)} fields in {category}")
                    # Process this category
                    processor.process_fields(
                        fields_to_process=list(category_fields.keys()),
                        cycle=cycle,
                        forecast_hour=fhr,
                        output_dir=cat_output_dir
                    )
                    
                    # Count successes (approximate based on existing files)
                    success_count = len(list(cat_output_dir.glob("*.png")))
                    total_success += success_count
                    total_attempted += len(category_fields)
                    
                    category_duration = time.time() - category_start
                    print(f"✅ {category}: {success_count}/{len(category_fields)} successful in {category_duration:.1f}s")
                    last_category_end = time.time()
                    print(f"📝 {category} completed at {last_category_end:.3f} ({time.strftime('%H:%M:%S')})")
                else:
                    print(f"⚠️ No fields found for category: {category}")
                    last_category_end = time.time()
        else:
            # Process all fields at once
            fields_to_process = [name for name, config in all_fields.items() 
                               if config.get('category') in categories_to_process]
            
            processor.process_fields(
                fields_to_process=fields_to_process,
                cycle=cycle,
                forecast_hour=fhr,
                output_dir=fhr_output_dir
            )
            
            # Count successes
            success_count = len(list(fhr_output_dir.glob("*.png")))
            total_success += success_count
            total_attempted += len(fields_to_process)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🎉 ALL PRODUCTS PROCESSING COMPLETE")
    print(f"=" * 60)
    print(f"Total maps generated: {total_success}")
    print(f"Total attempted: {total_attempted}")
    print(f"Success rate: {total_success/total_attempted*100:.1f}%")
    print(f"Output directory: {output_dir}")
    
    # Create index file
    create_product_index(output_dir, categories_to_process, forecast_hours)


def create_product_index(output_dir, categories, forecast_hours):
    """Create an index of all generated products"""
    index_file = output_dir / "product_index.txt"
    
    with open(index_file, 'w') as f:
        f.write("HRRR Product Index\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Categories: {', '.join(categories)}\n")
        f.write(f"Forecast Hours: {', '.join([f'F{h:02d}' for h in forecast_hours])}\n\n")
        
        # List all PNG files
        for fhr_dir in sorted(output_dir.glob("F*")):
            if fhr_dir.is_dir():
                f.write(f"\n{fhr_dir.name}:\n")
                
                # List by category if organized that way
                category_dirs = [d for d in fhr_dir.iterdir() if d.is_dir()]
                if category_dirs:
                    for cat_dir in sorted(category_dirs):
                        png_files = list(cat_dir.glob("*.png"))
                        f.write(f"  {cat_dir.name}: {len(png_files)} maps\n")
                        for png_file in sorted(png_files):
                            f.write(f"    {png_file.name}\n")
                else:
                    # Files directly in forecast hour directory
                    png_files = list(fhr_dir.glob("*.png"))
                    f.write(f"  {len(png_files)} maps\n")
                    for png_file in sorted(png_files):
                        f.write(f"    {png_file.name}\n")
    
    print(f"📋 Product index created: {index_file}")


def main():
    """Main function with command line options"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'all':
            # Process all products, all categories
            process_all_products(by_category=True)
            
        elif command == 'instability':
            # Process only instability products
            process_all_products(
                include_categories=['instability'],
                by_category=True
            )
            
        elif command == 'severe':
            # Process severe weather products
            process_all_products(
                include_categories=['severe', 'reflectivity'],
                by_category=True
            )
            
        elif command == 'surface':
            # Process surface products
            process_all_products(
                include_categories=['surface', 'precipitation'],
                by_category=True
            )
            
        elif command == 'smoke':
            # Process smoke products
            process_all_products(
                include_categories=['smoke'],
                by_category=True
            )
            
            
        elif command == 'forecast':
            # Process multiple forecast hours
            if len(sys.argv) > 2:
                try:
                    max_fhr = int(sys.argv[2])
                    forecast_hours = list(range(0, max_fhr + 1, 3))  # Every 3 hours
                    process_all_products(
                        forecast_hours=forecast_hours,
                        by_category=True
                    )
                except ValueError:
                    print("Invalid forecast hour. Use: python process_all_products.py forecast 12")
            else:
                print("Usage: python process_all_products.py forecast <max_hour>")
                
        elif command == 'quick':
            # Quick test with key products only
            key_fields = ['sbcape', 'reflectivity_comp', 't2m', 'wspd10m_max', 'precip_rate']
            processor = HRRRProcessor()
            processor.process_fields(
                fields_to_process=key_fields,
                output_dir=Path('./quick_test')
            )
            
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information"""
    print("🚀 HRRR All Products Processor")
    print("=" * 50)
    print("Usage:")
    print("  all                    - Process all products by category")
    print("  instability           - Process instability products only")  
    print("  severe                - Process severe weather products")
    print("  surface               - Process surface products")
    print("  smoke                 - Process smoke products")
    print("  forecast <hours>      - Process multiple forecast hours")
    print("  quick                 - Quick test with key products")
    print("\nExamples:")
    print("  python process_all_products.py all")
    print("  python process_all_products.py forecast 12")
    print("  python process_all_products.py quick")


if __name__ == '__main__':
    main()