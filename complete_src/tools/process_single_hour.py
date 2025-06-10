#!/usr/bin/env python3
"""
Single Hour Processor
Processes just one forecast hour to isolate memory usage
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path so we can import from the main project
sys.path.insert(0, str(Path(__file__).parent.parent))
from hrrr_processor_refactored import HRRRProcessor
from process_all_products import process_all_products

FILTER_FILE = Path(__file__).parent.parent / "custom_filters.json"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process single forecast hour')
    parser.add_argument('cycle', help='Cycle in YYYYMMDDHH format')
    parser.add_argument('forecast_hour', type=int, help='Forecast hour to process')
    parser.add_argument('--categories', help='Categories to process (comma-separated)')
    parser.add_argument('--fields', help='Specific fields to process (comma-separated)')
    parser.add_argument('--filter', help='Saved filter name')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--grib-dir', help='Directory containing GRIB files')
    parser.add_argument('--use-local-grib', action='store_true', help='Use GRIB files in output directory (no download)')

    args = parser.parse_args()

    try:
        # Parse categories
        include_categories = None
        if args.categories:
            include_categories = [cat.strip() for cat in args.categories.split(',')]

        # Parse fields
        fields = None
        if args.fields:
            fields = [f.strip() for f in args.fields.split(',')]

        # Load filter if specified
        if args.filter:
            if FILTER_FILE.exists():
                with open(FILTER_FILE, 'r') as f:
                    filters = json.load(f)
                if args.filter in filters:
                    filt = filters[args.filter]
                    if not fields:
                        fields = filt.get('fields')
                    if not include_categories:
                        include_categories = filt.get('categories')
                else:
                    print(f"Filter not found: {args.filter}")
            else:
                print("No filter file available")

        # Set output directory
        output_dir = args.output_dir
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'./single_hour_{args.cycle}_F{args.forecast_hour:02d}_{timestamp}'

        print(f"Processing {args.cycle} F{args.forecast_hour:02d}")
        print(f"Categories: {include_categories}")
        print(f"Output: {output_dir}")
        print(f"Use local GRIB: {args.use_local_grib}")

        if fields:
            # Directly process specified fields
            processor = HRRRProcessor()
            if args.use_local_grib:
                processor.use_local_grib = True
            if args.grib_dir:
                processor.grib_dir = Path(args.grib_dir)

            # Group fields by category so output stays organized
            from field_registry import FieldRegistry
            registry = FieldRegistry()
            cat_map = {}
            for fld in fields:
                cfg = registry.get_field(fld)
                if not cfg:
                    print(f"⚠️ Field not found: {fld}")
                    continue
                cat = cfg.get('category', 'misc')
                cat_map.setdefault(cat, []).append(fld)

            for cat, flist in cat_map.items():
                cat_dir = Path(output_dir) / cat
                cat_dir.mkdir(parents=True, exist_ok=True)
                processor.process_fields(
                    fields_to_process=flist,
                    cycle=args.cycle,
                    forecast_hour=args.forecast_hour,
                    output_dir=cat_dir
                )
        else:
            # Process via categories
            process_all_products(
                cycle=args.cycle,
                forecast_hours=[args.forecast_hour],
                output_dir=output_dir,
                by_category=True,
                include_categories=include_categories,
                use_local_grib=args.use_local_grib
            )

        print(f"✅ Completed F{args.forecast_hour:02d}")

    except Exception as e:
        print(f"❌ Error processing F{args.forecast_hour:02d}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
