import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.normalization import Normalizer
import json

class HRRRDataset(Dataset):
    """
    Returns (past_state → future_state) pairs at full 3 km resolution.
    Loads Zarr data directly without xarray to avoid dask issues.
    """
    def __init__(self, zarr_path: Path, variables, lead_hours: int,
                 stats_path: Path, sample_stride=1):
        # Open zarr store directly
        self.store = zarr.open(str(zarr_path), mode='r')
        self.vars = variables
        self.lead = lead_hours
        self.stride = sample_stride
        
        # Get time array
        self.times = self.store['time'][:]
        
        # Initialize normalizer
        self.norm = Normalizer(stats_path)
        
        # Determine valid indices (drop last N timesteps without a future target)
        self.valid_idx = np.arange(0, len(self.times) - self.lead, self.stride)
        
        # Cache dimensions
        first_var = self.vars[0]
        self.shape = self.store[first_var].shape[1:]  # spatial dimensions
        
        print(f"Dataset initialized:")
        print(f"  Variables: {self.vars}")
        print(f"  Time steps: {len(self.times)}")
        print(f"  Valid samples: {len(self.valid_idx)}")
        print(f"  Spatial shape: {self.shape}")

    def __len__(self):
        return len(self.valid_idx)

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
        
        return x, y


class HRRRDatasetWithCoords(HRRRDataset):
    """
    Extended version that also includes coordinate information.
    """
    def __init__(self, zarr_path: Path, variables, lead_hours: int,
                 stats_path: Path, sample_stride=1):
        super().__init__(zarr_path, variables, lead_hours, stats_path, sample_stride)
        
        # Load coordinate arrays
        self.latitude = self.store['latitude'][:]
        self.longitude = self.store['longitude'][:]

    def get_coordinates(self):
        """Return latitude and longitude arrays."""
        return self.latitude, self.longitude