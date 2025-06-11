#!/usr/bin/env python
"""Simple Zarr verification without xarray/dask."""
import zarr
import numpy as np
from pathlib import Path
import json

def verify_zarr_simple(zarr_path):
    """Verify Zarr dataset using only zarr library."""
    print(f"Verifying Zarr dataset at: {zarr_path}\n")
    
    # Open zarr store
    store = zarr.open(zarr_path, mode='r')
    
    print("Zarr Arrays:")
    print("="*50)
    
    # List all arrays
    arrays = {}
    for name, arr in store.arrays():
        arrays[name] = arr
        print(f"\n{name}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Chunks: {arr.chunks}")
        print(f"  Compressor: {arr.compressor}")
        
        # Get sample statistics (just from first chunk to avoid loading all data)
        if arr.size > 0:
            # Load a small slice
            if len(arr.shape) == 3:  # time, y, x
                sample = arr[0, :100, :100]
            elif len(arr.shape) == 2:  # y, x
                sample = arr[:100, :100]
            else:
                sample = arr[:100]
            
            print(f"  Sample min: {np.min(sample):.3f}")
            print(f"  Sample max: {np.max(sample):.3f}")
            print(f"  Sample mean: {np.mean(sample):.3f}")
    
    # Check consolidated metadata
    print("\nConsolidated Metadata:")
    print("="*50)
    try:
        metadata = zarr.open_consolidated(zarr_path)
        print("✓ Consolidated metadata found")
    except:
        print("✗ No consolidated metadata")
    
    # Check .zmetadata file
    zmetadata_path = Path(zarr_path) / '.zmetadata'
    if zmetadata_path.exists():
        with open(zmetadata_path, 'r') as f:
            zmeta = json.load(f)
        print(f"\n.zmetadata keys: {list(zmeta.keys())}")
        if 'metadata' in zmeta:
            print(f"Number of arrays: {len([k for k in zmeta['metadata'].keys() if '.zarray' in k])}")
    
    return arrays

if __name__ == "__main__":
    verify_zarr_simple("data/zarr/training_data/hrrr.zarr")