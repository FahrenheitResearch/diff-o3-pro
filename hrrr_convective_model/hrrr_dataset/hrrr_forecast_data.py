import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.normalization import Normalizer
import random

class HRRRForecastDataset(Dataset):
    """
    Dataset for HRRR forecast sequences.
    Returns (input_state, target_state, timestamps) pairs where:
    - input_state: forecast at time t
    - target_state: forecast at time t+1
    - timestamps: hour information for temporal encoding
    
    The zarr store contains sequences where odd indices are inputs
    and even indices are targets.
    """
    def __init__(self, zarr_path: Path, variables, stats_path: Path):
        # Open zarr store directly
        self.store = zarr.open(str(zarr_path), mode='r')
        self.vars = variables
        
        # Initialize normalizer
        self.norm = Normalizer(stats_path)
        
        # Get total timesteps and identify input/target pairs
        total_timesteps = self.store[variables[0]].shape[0]
        
        # Build list of (input_idx, target_idx) pairs
        self.sequences = []
        for i in range(0, total_timesteps - 1, 2):
            # Check if this is a valid input->target pair
            if (self.store.attrs.get(f'time_{i}_type') == 'input' and 
                self.store.attrs.get(f'time_{i+1}_type') == 'target'):
                self.sequences.append((i, i+1))
        
        # Cache dimensions
        first_var = self.vars[0]
        self.shape = self.store[first_var].shape[1:]  # spatial dimensions
        
        print(f"Forecast Dataset initialized:")
        print(f"  Variables: {self.vars}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Training sequences: {len(self.sequences)}")
        print(f"  Spatial shape: {self.shape}")
        
        # Print some sequence info
        if len(self.sequences) > 0:
            first_seq = self.sequences[0]
            print(f"  First sequence: F{self.store.attrs.get(f'time_{first_seq[0]}_fh', 'unknown')}" + 
                  f" -> F{self.store.attrs.get(f'time_{first_seq[1]}_fh', 'unknown')}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_idx, target_idx = self.sequences[idx]
        
        # Get forecast hours for temporal encoding
        input_fh = self.store.attrs.get(f'time_{input_idx}_fh', 0)
        target_fh = self.store.attrs.get(f'time_{target_idx}_fh', 1)
        
        # Extract initialization time and compute hour of day
        init_time_str = self.store.attrs.get(f'time_{input_idx}_init', '')
        try:
            # Parse initialization time to get hour of day
            # Format: "2025-06-11T20:00:00"
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
            input_raw = self.store[var][input_idx, :, :]
            input_norm = self.norm.encode(input_raw, var)
            input_data.append(input_norm)
            
            # Load and normalize target
            target_raw = self.store[var][target_idx, :, :]
            target_norm = self.norm.encode(target_raw, var)
            target_data.append(target_norm)
        
        # Stack variables along channel dimension
        input_tensor = torch.from_numpy(np.stack(input_data, axis=0)).float()
        target_tensor = torch.from_numpy(np.stack(target_data, axis=0)).float()
        
        # Create timestamp tensors for temporal encoding
        # Shape: [2] containing [input_hour, target_hour]
        timestamps = torch.tensor([input_hour, target_hour], dtype=torch.float32)
        
        return input_tensor, target_tensor, timestamps
    
    def get_forecast_info(self, idx):
        """Get metadata about a specific sequence."""
        input_idx, target_idx = self.sequences[idx]
        return {
            'init_time': self.store.attrs.get(f'time_{input_idx}_init', 'unknown'),
            'input_fh': self.store.attrs.get(f'time_{input_idx}_fh', -1),
            'target_fh': self.store.attrs.get(f'time_{target_idx}_fh', -1),
        }