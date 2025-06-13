import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.normalization import Normalizer
import random

class HRRRDatasetFixed(Dataset):
    """
    Fixed HRRR dataset that:
    1. Returns real timestamps for temporal encoding
    2. Supports variable lead times (default 1 hour)
    3. Includes data augmentation options
    """
    def __init__(self, zarr_path: Path, variables, lead_hours: int,
                 stats_path: Path, sample_stride=1, augment=False,
                 return_timestamps=True, epoch_start_hours=0):
        # Open zarr store directly
        self.store = zarr.open(str(zarr_path), mode='r')
        self.vars = variables
        self.lead = lead_hours
        self.stride = sample_stride
        self.augment = augment
        self.return_timestamps = return_timestamps
        self.epoch_start_hours = epoch_start_hours  # Hours since epoch for time[0]
        
        # Get time array (assumed to be hours since start)
        self.times = self.store['time'][:]
        
        # Initialize normalizer
        self.norm = Normalizer(stats_path)
        
        # Determine valid indices
        self.valid_idx = np.arange(0, len(self.times) - self.lead, self.stride)
        
        # Cache dimensions
        first_var = self.vars[0]
        self.shape = self.store[first_var].shape[1:]  # spatial dimensions
        
        print(f"Dataset initialized:")
        print(f"  Variables: {self.vars}")
        print(f"  Time steps: {len(self.times)}")
        print(f"  Valid samples: {len(self.valid_idx)}")
        print(f"  Spatial shape: {self.shape}")
        print(f"  Lead time: {self.lead} hours")
        print(f"  Augmentation: {self.augment}")

    def __len__(self):
        return len(self.valid_idx)
    
    def _augment(self, x, y):
        """Apply data augmentation (same transforms to x and y)."""
        # Random horizontal flip
        if random.random() > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        
        # Random vertical flip  
        if random.random() > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        
        # Small random noise (only during training)
        if random.random() > 0.5:
            noise_scale = 0.01
            x = x + torch.randn_like(x) * noise_scale
            
        return x, y

    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        past = []
        future = []
        
        for k in self.vars:
            # Get data for current and future time
            past_arr = self.store[k][t].astype(np.float32)
            future_arr = self.store[k][t + self.lead].astype(np.float32)
            
            # Apply normalization
            past.append(self.norm.encode(past_arr, k))
            future.append(self.norm.encode(future_arr, k))
        
        # Stack to create [C, H, W] tensors
        x = torch.tensor(np.stack(past, axis=0))
        y = torch.tensor(np.stack(future, axis=0))
        
        # Apply augmentation if enabled
        if self.augment:
            x, y = self._augment(x, y)
        
        if self.return_timestamps:
            # Return actual timestamp (hours since epoch)
            # Assuming times array contains hours since data start
            timestamp_x = self.epoch_start_hours + self.times[t]
            timestamp_y = self.epoch_start_hours + self.times[t + self.lead]
            return x, y, timestamp_x, timestamp_y
        else:
            return x, y

def compute_normalization_stats_fixed(dataset):
    """Compute mean and std for each variable using Welford's algorithm."""
    n_vars = len(dataset.vars)
    n_points = dataset.shape[0] * dataset.shape[1]
    
    # Initialize statistics
    count = 0
    mean = np.zeros(n_vars)
    M2 = np.zeros(n_vars)
    
    print("Computing normalization statistics...")
    # Sample subset of data for efficiency
    sample_size = min(len(dataset), 1000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        if dataset.return_timestamps:
            x, _, _, _ = dataset[idx]
        else:
            x, _ = dataset[idx]
        
        # Update statistics for each variable
        for i in range(n_vars):
            data = x[i].numpy().flatten()
            for value in data:
                count += 1
                delta = value - mean[i]
                mean[i] += delta / count
                delta2 = value - mean[i]
                M2[i] += delta * delta2
    
    # Compute standard deviation
    variance = M2 / (count - 1)
    std = np.sqrt(variance)
    
    # Ensure no zero std
    std = np.maximum(std, 1e-6)
    
    return mean, std