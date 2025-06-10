#!/usr/bin/env python3
"""
Simple timing script for individual field generation
Times how long each field takes using process_single_hour.py
"""

import subprocess
import time
import sys
import json
from pathlib import Path

def time_field(cycle, forecast_hour, output_dir, field_name):
    """Time a single field processing"""
    cmd = [
        'python3', './process_single_hour.py',
        cycle, str(forecast_hour),
        '--output-dir', output_dir,
        '--use-local-grib',
        '--fields', field_name
    ]
    
    print(f"⏱️  {field_name:20s}", end="", flush=True)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout per field
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f" ✅ {duration:6.2f}s")
            return {'duration': duration, 'status': 'success'}
        else:
            print(f" ❌ {duration:6.2f}s (EXIT: {result.returncode})")
            print(f"    Error: {result.stderr[:100]}")
            return {'duration': duration, 'status': 'failed', 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f" ⏰ {duration:6.2f}s (TIMEOUT)")
        return {'duration': duration, 'status': 'timeout'}
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f" ❌ {duration:6.2f}s (ERROR: {str(e)[:50]})")
        return {'duration': duration, 'status': 'failed', 'error': str(e)}

def main():
    if len(sys.argv) < 3:
        print("Usage: python time_fields.py <cycle> <forecast_hour> [field1,field2,...]")
        print("Example: python time_fields.py 2025060212 6")
        print("Example: python time_fields.py 2025060212 6 t2m,sbcape,vtp")
        sys.exit(1)
    
    cycle = sys.argv[1]
    forecast_hour = int(sys.argv[2])
    
    # Auto-generate output dir
    date_str = cycle[:8]
    hour_str = f"{cycle[8:10]}z"
    output_dir = f"outputs/hrrr/{date_str}/{hour_str}/F{forecast_hour:02d}"
    
    if len(sys.argv) > 3:
        fields = sys.argv[3].split(',')
    else:
        # Default test fields
        fields = [
            't2m', 'sbcape', 'mlcape', 'wind_speed_10m', 
            'reflectivity_comp', 'precip_rate', 'hail',
            'scp', 'stp', 'ehi', 'wind_shear_06km',
            'lapse_rate_03km', 'vtp'
        ]
    
    print(f"🕐 Timing field processing for {cycle} F{forecast_hour:02d}")
    print(f"📁 Output: {output_dir}")
    print(f"🧪 Testing {len(fields)} fields")
    print("=" * 60)
    
    timing_results = {}
    
    for field_name in fields:
        result = time_field(cycle, forecast_hour, output_dir, field_name)
        timing_results[field_name] = result
    
    print("\n📊 TIMING SUMMARY")
    print("=" * 60)
    
    # Sort by duration
    sorted_results = sorted(timing_results.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    total_time = sum(r['duration'] for r in timing_results.values())
    successful = sum(1 for r in timing_results.values() if r['status'] == 'success')
    timeouts = sum(1 for r in timing_results.values() if r['status'] == 'timeout')
    
    for field_name, result in sorted_results:
        status_icon = "✅" if result['status'] == 'success' else "⏰" if result['status'] == 'timeout' else "❌"
        duration = result['duration']
        print(f"{status_icon} {field_name:20s} {duration:6.2f}s")
    
    print("-" * 60)
    print(f"📈 Total time: {total_time:.2f}s")
    print(f"📊 Success: {successful}/{len(fields)} ({successful/len(fields)*100:.1f}%)")
    if timeouts > 0:
        print(f"⏰ Timeouts: {timeouts}")
    print(f"⚡ Average time: {total_time/len(fields):.2f}s per field")
    
    # Identify slow fields
    slow_threshold = 10.0  # seconds
    slow_fields = [name for name, result in timing_results.items() 
                   if result['duration'] > slow_threshold and result['status'] in ['success', 'timeout']]
    
    if slow_fields:
        print(f"\n🐌 Slow fields (>{slow_threshold}s):")
        for field in slow_fields:
            duration = timing_results[field]['duration']
            status = timing_results[field]['status']
            print(f"   {field}: {duration:.2f}s ({status})")
    
    # Save detailed results
    results_file = Path(output_dir) / f"timing_results_{cycle}_F{forecast_hour:02d}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Timing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during timing: {e}")
        import traceback
        traceback.print_exc()