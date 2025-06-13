#!/usr/bin/env python3
"""Real-time training monitor that updates plots every N seconds."""

import time
import subprocess
import sys
from pathlib import Path

def monitor_training(update_interval=30):
    """Monitor training and update plots periodically."""
    print(f"Starting training monitor (updating every {update_interval} seconds)")
    print("Press Ctrl+C to stop monitoring")
    
    plot_script = Path('plot_training_progress.py')
    if not plot_script.exists():
        print("Error: plot_training_progress.py not found!")
        return
    
    iteration = 0
    while True:
        try:
            # Clear screen for clean display
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            
            print(f"=== Training Monitor - Update #{iteration} ===")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check if training is still running
            result = subprocess.run(['pgrep', '-f', 'train_deterministic_optimized.py'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("✓ Training is running (PID: {})".format(result.stdout.strip()))
            else:
                print("⚠ Training process not found!")
            
            # Update plots
            print("\nUpdating plots...")
            subprocess.run([sys.executable, 'plot_training_progress.py'])
            
            # Show latest metrics from log
            print("\nLatest training metrics:")
            subprocess.run(['tail', '-n', '20', 'training.log'])
            
            print(f"\nNext update in {update_interval} seconds...")
            time.sleep(update_interval)
            iteration += 1
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Check if custom interval is provided
    import argparse
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Update interval in seconds (default: 30)')
    args = parser.parse_args()
    
    monitor_training(args.interval)