#!/usr/bin/env python3
"""
Quick timing script that shows which step is hanging
"""

import subprocess
import time
import sys
import signal
from pathlib import Path

def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")

def quick_time_field(cycle, forecast_hour, field_name, timeout_sec=30):
    """Quick test of field timing with verbose output"""
    
    output_dir = f"outputs/hrrr/{cycle[:8]}/{cycle[8:10]}z/F{forecast_hour:02d}"
    
    cmd = [
        'python3', './process_single_hour.py',
        cycle, str(forecast_hour),
        '--output-dir', output_dir,
        '--use-local-grib',
        '--fields', field_name
    ]
    
    print(f"🧪 Testing {field_name} (timeout: {timeout_sec}s)")
    print(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line to see where it hangs
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                elapsed = time.time() - start_time
                print(f"   [{elapsed:5.1f}s] {output.strip()}")
        
        signal.alarm(0)  # Cancel timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode == 0:
            print(f"✅ {field_name} completed in {duration:.2f}s")
        else:
            stderr = process.stderr.read()
            print(f"❌ {field_name} failed in {duration:.2f}s")
            if stderr:
                print(f"   Error: {stderr[:200]}")
        
        return duration
        
    except TimeoutError:
        if 'process' in locals():
            process.terminate()
            process.wait()
        signal.alarm(0)
        duration = time.time() - start_time
        print(f"⏰ {field_name} timed out after {duration:.2f}s")
        return duration
        
    except Exception as e:
        signal.alarm(0)
        duration = time.time() - start_time
        print(f"❌ {field_name} error after {duration:.2f}s: {e}")
        return duration

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python quick_timing.py <cycle> <forecast_hour> <field_name> [timeout_sec]")
        print("Example: python quick_timing.py 2025060212 6 t2m 30")
        sys.exit(1)
    
    cycle = sys.argv[1]
    forecast_hour = int(sys.argv[2])
    field_name = sys.argv[3]
    timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    
    try:
        duration = quick_time_field(cycle, forecast_hour, field_name, timeout_sec)
        print(f"\n⏱️  Total duration: {duration:.2f}s")
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()