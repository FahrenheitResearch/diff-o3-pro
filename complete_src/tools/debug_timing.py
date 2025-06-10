#!/usr/bin/env python3
"""
Debug script to time individual map generation
Logs how long each field takes to process
"""

import time
import sys
from pathlib import Path
from hrrr_processor_refactored import HRRRProcessor
from field_registry import FieldRegistry
import json

def time_field_processing(cycle, forecast_hour, output_dir, fields=None):
    """Time individual field processing"""
    
    print(f"🕐 Timing field processing for {cycle} F{forecast_hour:02d}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)
    
    # Get field registry
    registry = FieldRegistry()
    all_fields = registry.load_all_fields()
    
    if fields:
        test_fields = [f for f in fields if f in all_fields]
    else:
        # Test a subset of common fields
        test_fields = [
            't2m', 'sbcape', 'mlcape', 'wind_speed_10m', 
            'reflectivity_comp', 'precip_rate', 'hail',
            'scp', 'stp', 'ehi', 'wind_shear_06km',
            'lapse_rate_03km', 'vtp'
        ]
        test_fields = [f for f in test_fields if f in all_fields]
    
    print(f"🧪 Testing {len(test_fields)} fields:")
    print(f"   {', '.join(test_fields)}")
    print()
    
    # Initialize processor
    processor = HRRRProcessor()
    
    timing_results = {}
    
    for field_name in test_fields:
        print(f"⏱️  Processing {field_name}...", end="", flush=True)
        
        start_time = time.time()
        
        try:
            # Run single field processing
            result = processor.process_fields(
                fields_to_process=[field_name],
                cycle=cycle,
                forecast_hour=forecast_hour,
                output_dir=output_dir
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            timing_results[field_name] = {
                'duration': duration,
                'status': 'success'
            }
            
            print(f" ✅ {duration:.2f}s")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            timing_results[field_name] = {
                'duration': duration,
                'status': 'failed',
                'error': str(e)
            }
            
            print(f" ❌ {duration:.2f}s (ERROR: {str(e)[:50]})")
    
    print()
    print("📊 TIMING SUMMARY")
    print("=" * 60)
    
    # Sort by duration
    sorted_results = sorted(timing_results.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    total_time = sum(r['duration'] for r in timing_results.values())
    successful = sum(1 for r in timing_results.values() if r['status'] == 'success')
    
    for field_name, result in sorted_results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        duration = result['duration']
        print(f"{status_icon} {field_name:20s} {duration:6.2f}s")
    
    print("-" * 60)
    print(f"📈 Total time: {total_time:.2f}s")
    print(f"📊 Success rate: {successful}/{len(test_fields)} ({successful/len(test_fields)*100:.1f}%)")
    print(f"⚡ Average time: {total_time/len(test_fields):.2f}s per field")
    
    # Identify slow fields
    slow_threshold = 10.0  # seconds
    slow_fields = [name for name, result in timing_results.items() 
                   if result['duration'] > slow_threshold and result['status'] == 'success']
    
    if slow_fields:
        print(f"\n🐌 Slow fields (>{slow_threshold}s):")
        for field in slow_fields:
            duration = timing_results[field]['duration']
            print(f"   {field}: {duration:.2f}s")
    
    # Save detailed results
    results_file = Path(output_dir) / f"timing_results_{cycle}_F{forecast_hour:02d}.json"
    with open(results_file, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return timing_results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_timing.py <cycle> <forecast_hour> [output_dir] [field1,field2,...]")
        print("Example: python debug_timing.py 2025060212 6")
        print("Example: python debug_timing.py 2025060212 6 /path/to/output t2m,sbcape,vtp")
        sys.exit(1)
    
    cycle = sys.argv[1]
    forecast_hour = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    else:
        # Auto-generate output dir
        date_str = cycle[:8]
        hour_str = f"{cycle[8:10]}z"
        output_dir = f"outputs/hrrr/{date_str}/{hour_str}/F{forecast_hour:02d}"
    
    if len(sys.argv) > 4:
        fields = sys.argv[4].split(',')
    else:
        fields = None
    
    try:
        timing_results = time_field_processing(cycle, forecast_hour, output_dir, fields)
    except KeyboardInterrupt:
        print("\n\n⏹️  Timing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during timing: {e}")
        import traceback
        traceback.print_exc()