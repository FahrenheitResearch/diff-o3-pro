#!/usr/bin/env python
"""
Streaming HRRR preprocessor that works with partially downloaded data.
Monitors the download manifest and processes cycles as they become ready.
"""
from pathlib import Path
import argparse
import json
import xarray as xr
import zarr
import numpy as np
from datetime import datetime
import re
import warnings
import time
import threading
warnings.filterwarnings("ignore")

from preprocess_hrrr_forecast import HRRR_VARS, open_hrrr_grib, extract_time_info_from_filename


class StreamingProcessor:
    def __init__(self, src_path, out_path, manifest_path):
        self.src_path = Path(src_path)
        self.out_path = Path(out_path)
        self.manifest_path = Path(manifest_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        
        # Track processed cycles
        self.processed_cycles = set()
        self.processing_state_file = self.out_path / 'processing_state.json'
        self._load_processing_state()
        
        # Initialize zarr store
        self.zarr_path = self.out_path / "hrrr.zarr"
        self.store = None
        self.time_idx = 0
        self.dimensions_set = False
        
    def _load_processing_state(self):
        """Load the state of what we've already processed."""
        if self.processing_state_file.exists():
            with open(self.processing_state_file, 'r') as f:
                state = json.load(f)
                self.processed_cycles = set(state.get('processed_cycles', []))
                self.time_idx = state.get('time_idx', 0)
    
    def _save_processing_state(self):
        """Save the current processing state."""
        state = {
            'processed_cycles': list(self.processed_cycles),
            'time_idx': self.time_idx,
            'last_update': datetime.now().isoformat()
        }
        with open(self.processing_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_manifest(self):
        """Load the download manifest."""
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def _get_ready_cycles(self, manifest, min_hours=2):
        """Get cycles ready for processing."""
        ready = []
        for cycle_key, cycle_data in manifest.get('partial_cycles', {}).items():
            if cycle_key in self.processed_cycles:
                continue
            
            # Check for consecutive hours starting from 0
            hours = cycle_data.get('forecast_hours', {})
            consecutive = 0
            files = []
            
            for h in range(19):  # F00-F18
                if str(h) in hours:
                    consecutive += 1
                    files.append(Path(hours[str(h)]))
                else:
                    break
            
            if consecutive >= min_hours:
                ready.append((cycle_key, files[:consecutive]))
        
        return ready
    
    def _initialize_zarr(self, first_ds):
        """Initialize the zarr store with proper dimensions."""
        if self.dimensions_set:
            return
        
        # Get dimensions from first dataset
        first_var = list(first_ds.data_vars.values())[0]
        ny, nx = first_var.shape
        
        # We'll allocate space incrementally
        initial_time_size = 1000  # Start with space for 1000 timesteps
        
        print(f"Initializing Zarr store:")
        print(f"  Spatial dimensions: {ny} x {nx}")
        print(f"  Initial time allocation: {initial_time_size}")
        
        self.store = zarr.open_group(str(self.zarr_path), mode='a')
        
        # Create or resize arrays for each variable
        for var_name in HRRR_VARS.keys():
            if var_name not in self.store:
                self.store.create_dataset(
                    var_name,
                    shape=(initial_time_size, ny, nx),
                    chunks=(1, ny, nx),
                    dtype='float32',
                    fill_value=np.nan
                )
        
        # Store coordinates if not present
        if 'latitude' not in self.store and 'latitude' in first_ds.coords:
            lat = first_ds.latitude.values
            lon = first_ds.longitude.values
            self.store.create_dataset('latitude', data=lat, dtype='float32')
            self.store.create_dataset('longitude', data=lon, dtype='float32')
        
        # Create time array if not present
        if 'time' not in self.store:
            self.store.create_dataset('time', shape=(initial_time_size,), dtype='int64')
        
        self.dimensions_set = True
    
    def _resize_if_needed(self, required_size):
        """Resize zarr arrays if we're running out of space."""
        current_size = self.store[list(HRRR_VARS.keys())[0]].shape[0]
        
        if required_size > current_size:
            new_size = max(required_size, current_size * 2)  # Double the size
            print(f"Resizing arrays from {current_size} to {new_size} timesteps...")
            
            for var_name in HRRR_VARS.keys():
                if var_name in self.store:
                    arr = self.store[var_name]
                    arr.resize(new_size, arr.shape[1], arr.shape[2])
            
            self.store['time'].resize(new_size)
    
    def process_cycle(self, cycle_key, files):
        """Process a single forecast cycle."""
        print(f"\nProcessing cycle: {cycle_key} with {len(files)} files")
        
        # Sort files by forecast hour
        sorted_files = sorted(files, key=lambda f: int(re.search(r'wrfprsf(\d{2})', f.name).group(1)))
        
        # Load all files for this cycle
        processed_hours = {}
        for filepath in sorted_files:
            result = open_hrrr_grib(filepath)
            if result is None:
                continue
            
            combined, init_time, forecast_hour = result
            
            # Initialize zarr on first successful read
            if not self.dimensions_set:
                self._initialize_zarr(combined)
            
            processed_hours[forecast_hour] = combined
        
        # Create sequences and write to zarr
        sequences_added = 0
        for fh in sorted(processed_hours.keys())[:-1]:
            if fh + 1 in processed_hours:
                # Ensure we have space
                self._resize_if_needed(self.time_idx + 2)
                
                # Write input state
                for var_name in HRRR_VARS.keys():
                    if var_name in processed_hours[fh].data_vars:
                        input_data = processed_hours[fh][var_name].values
                        self.store[var_name][self.time_idx, :, :] = input_data
                
                # Store metadata
                self.store.attrs[f'time_{self.time_idx}_init'] = processed_hours[fh].attrs['init_time']
                self.store.attrs[f'time_{self.time_idx}_fh'] = fh
                self.store.attrs[f'time_{self.time_idx}_type'] = 'input'
                
                self.time_idx += 1
                
                # Write target state
                for var_name in HRRR_VARS.keys():
                    if var_name in processed_hours[fh + 1].data_vars:
                        target_data = processed_hours[fh + 1][var_name].values
                        self.store[var_name][self.time_idx, :, :] = target_data
                
                self.store.attrs[f'time_{self.time_idx}_init'] = processed_hours[fh + 1].attrs['init_time']
                self.store.attrs[f'time_{self.time_idx}_fh'] = fh + 1
                self.store.attrs[f'time_{self.time_idx}_type'] = 'target'
                
                self.time_idx += 1
                sequences_added += 1
        
        # Update metadata
        self.store.attrs['current_sequences'] = self.time_idx // 2
        self.store.attrs['last_update'] = datetime.now().isoformat()
        
        print(f"  Added {sequences_added} sequences (total: {self.time_idx // 2})")
        
        # Mark as processed
        self.processed_cycles.add(cycle_key)
        self._save_processing_state()
    
    def run(self, check_interval=30, min_hours=2):
        """Run the streaming processor."""
        print(f"Starting streaming processor")
        print(f"  Source: {self.src_path}")
        print(f"  Output: {self.zarr_path}")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Check interval: {check_interval}s")
        print(f"  Min hours to process: {min_hours}")
        
        # If zarr exists, load existing store
        if self.zarr_path.exists():
            self.store = zarr.open_group(str(self.zarr_path), mode='a')
            self.dimensions_set = True
        
        while True:
            # Load manifest
            manifest = self._load_manifest()
            if manifest is None:
                print("Waiting for download manifest...")
                time.sleep(check_interval)
                continue
            
            # Check for ready cycles
            ready_cycles = self._get_ready_cycles(manifest, min_hours)
            
            if ready_cycles:
                print(f"\nFound {len(ready_cycles)} cycles ready for processing")
                for cycle_key, files in ready_cycles:
                    self.process_cycle(cycle_key, files)
            
            # Check if we're done
            total_expected = manifest.get('total_expected_cycles', 0)
            if len(self.processed_cycles) >= total_expected and total_expected > 0:
                print(f"\nAll {total_expected} cycles processed!")
                break
            
            # Status update
            print(f"\rProcessed {len(self.processed_cycles)}/{total_expected} cycles, "
                  f"{self.time_idx // 2} sequences total", end='', flush=True)
            
            time.sleep(check_interval)
        
        # Final statistics
        print(f"\n\nProcessing complete!")
        print(f"  Total cycles: {len(self.processed_cycles)}")
        print(f"  Total sequences: {self.time_idx // 2}")
        print(f"  Output: {self.zarr_path}")


def main():
    parser = argparse.ArgumentParser(description="Streaming HRRR preprocessor")
    parser.add_argument("--src", type=str, required=True, help="Source directory with GRIB2 files")
    parser.add_argument("--out", type=str, required=True, help="Output directory for Zarr store")
    parser.add_argument("--manifest", type=str, help="Download manifest path")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--min-hours", type=int, default=2, help="Minimum consecutive hours to process")
    
    args = parser.parse_args()
    
    # Default manifest path
    if args.manifest is None:
        args.manifest = Path(args.src) / 'download_manifest.json'
    
    processor = StreamingProcessor(args.src, args.out, args.manifest)
    processor.run(check_interval=args.check_interval, min_hours=args.min_hours)


if __name__ == "__main__":
    main()