import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.normalization import Normalizer
import json
import threading
import time

class HRRRStreamingDataset(Dataset):
    """
    Dataset that works with streaming zarr data that's being written to.
    Periodically checks for new sequences and updates its length.
    """
    def __init__(self, zarr_path: Path, variables, stats_path: Path, 
                 check_interval=60, min_sequences=10):
        self.zarr_path = zarr_path
        self.vars = variables
        self.check_interval = check_interval
        self.min_sequences = min_sequences
        
        # Initialize normalizer
        self.norm = Normalizer(stats_path)
        
        # Thread-safe sequence tracking
        self.lock = threading.RLock()
        self.sequences = []
        self.last_check = 0
        self.last_sequence_count = 0
        
        # Wait for minimum data
        print(f"Waiting for zarr store at {zarr_path}...")
        while not zarr_path.exists():
            time.sleep(5)
        
        # Open zarr store
        self.store = zarr.open(str(zarr_path), mode='r')
        
        # Initial load
        self._update_sequences()
        
        # Wait for minimum sequences
        while len(self.sequences) < min_sequences:
            print(f"Waiting for minimum sequences ({len(self.sequences)}/{min_sequences})...")
            time.sleep(10)
            self._update_sequences()
        
        # Cache dimensions
        first_var = self.vars[0]
        self.shape = self.store[first_var].shape[1:]  # spatial dimensions
        
        print(f"Streaming Dataset initialized:")
        print(f"  Variables: {self.vars}")
        print(f"  Initial sequences: {len(self.sequences)}")
        print(f"  Spatial shape: {self.shape}")
        print(f"  Will check for updates every {check_interval}s")
        
        # Start background updater thread
        self.updater_thread = threading.Thread(target=self._background_updater, daemon=True)
        self.updater_thread.start()
    
    def _update_sequences(self):
        """Update the list of available sequences."""
        try:
            # Re-open store to get latest data
            store = zarr.open(str(self.zarr_path), mode='r')
            
            # Get current sequence count from metadata
            current_sequences = store.attrs.get('current_sequences', 0)
            
            if current_sequences > self.last_sequence_count:
                with self.lock:
                    # Clear and rebuild sequence list
                    self.sequences = []
                    
                    # Check each pair of timesteps
                    total_timesteps = store[self.vars[0]].shape[0]
                    for i in range(0, min(current_sequences * 2, total_timesteps - 1), 2):
                        # Verify this is a valid sequence
                        if (store.attrs.get(f'time_{i}_type') == 'input' and 
                            store.attrs.get(f'time_{i+1}_type') == 'target'):
                            self.sequences.append((i, i+1))
                    
                    self.last_sequence_count = current_sequences
                    
                    if len(self.sequences) > self.last_sequence_count:
                        print(f"\n[Dataset Update] New sequences available: {len(self.sequences)} total")
            
            self.last_check = time.time()
            
        except Exception as e:
            print(f"Error updating sequences: {e}")
    
    def _background_updater(self):
        """Background thread that checks for new data."""
        while True:
            time.sleep(self.check_interval)
            self._update_sequences()
    
    def __len__(self):
        with self.lock:
            return len(self.sequences)
    
    def __getitem__(self, idx):
        with self.lock:
            if idx >= len(self.sequences):
                # Force update and retry
                self._update_sequences()
                if idx >= len(self.sequences):
                    raise IndexError(f"Index {idx} out of range for {len(self.sequences)} sequences")
            
            input_idx, target_idx = self.sequences[idx]
        
        # Re-open store for each access to ensure fresh data
        store = zarr.open(str(self.zarr_path), mode='r')
        
        # Get forecast hours for temporal encoding
        input_fh = store.attrs.get(f'time_{input_idx}_fh', 0)
        target_fh = store.attrs.get(f'time_{target_idx}_fh', 1)
        
        # Extract initialization time and compute hour of day
        init_time_str = store.attrs.get(f'time_{input_idx}_init', '')
        try:
            init_hour = int(init_time_str.split('T')[1].split(':')[0])
        except:
            init_hour = 0
        
        # Compute actual valid time hours
        input_hour = (init_hour + input_fh) % 24
        target_hour = (init_hour + target_fh) % 24
        
        # Load data for all variables
        input_data = []
        target_data = []
        
        for var in self.vars:
            # Load and normalize input
            input_raw = store[var][input_idx, :, :]
            input_norm = self.norm.normalize(input_raw, var)
            input_data.append(input_norm)
            
            # Load and normalize target
            target_raw = store[var][target_idx, :, :]
            target_norm = self.norm.normalize(target_raw, var)
            target_data.append(target_norm)
        
        # Stack variables along channel dimension
        input_tensor = torch.from_numpy(np.stack(input_data, axis=0)).float()
        target_tensor = torch.from_numpy(np.stack(target_data, axis=0)).float()
        
        # Create timestamp tensors
        timestamps = torch.tensor([input_hour, target_hour], dtype=torch.float32)
        
        return input_tensor, target_tensor, timestamps
    
    def get_current_stats(self):
        """Get current dataset statistics."""
        with self.lock:
            return {
                'sequences': len(self.sequences),
                'last_update': time.time() - self.last_check,
                'store_sequences': self.last_sequence_count
            }