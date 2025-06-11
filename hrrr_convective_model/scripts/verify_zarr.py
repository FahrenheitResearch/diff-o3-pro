#!/usr/bin/env python
"""Verify the Zarr output from HRRR preprocessing."""
import xarray as xr
import zarr
from pathlib import Path
import sys

def verify_zarr(zarr_path):
    """Verify Zarr dataset contents."""
    print(f"Verifying Zarr dataset at: {zarr_path}\n")
    
    # Open with xarray
    ds = xr.open_zarr(zarr_path)
    
    print("Dataset Info:")
    print("="*50)
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")
    
    print("\nVariable Details:")
    print("="*50)
    for var in ds.data_vars:
        data = ds[var]
        print(f"\n{var}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dimensions: {data.dims}")
        print(f"  Data type: {data.dtype}")
        print(f"  Min: {float(data.min().compute()):.3f}")
        print(f"  Max: {float(data.max().compute()):.3f}")
        print(f"  Mean: {float(data.mean().compute()):.3f}")
        if hasattr(data, 'attrs') and data.attrs:
            print(f"  Attributes: {data.attrs}")
    
    print("\nTime Information:")
    print("="*50)
    times = ds.time.values
    print(f"Number of time steps: {len(times)}")
    print(f"First time: {times[0]}")
    print(f"Last time: {times[-1]}")
    
    print("\nSpatial Information:")
    print("="*50)
    if 'latitude' in ds.coords:
        lat = ds.latitude
        lon = ds.longitude
        print(f"Latitude range: {float(lat.min()):.2f} to {float(lat.max()):.2f}")
        print(f"Longitude range: {float(lon.min()):.2f} to {float(lon.max()):.2f}")
    
    # Check Zarr store directly
    print("\nZarr Store Info:")
    print("="*50)
    store = zarr.open(zarr_path, mode='r')
    print(f"Store type: {type(store)}")
    print(f"Arrays in store: {list(store.arrays())}")
    
    # Check chunk sizes
    print("\nChunk Information:")
    print("="*50)
    for var in ds.data_vars:
        chunks = ds[var].chunks
        print(f"{var}: chunks = {chunks}")
    
    return ds

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    else:
        zarr_path = "data/zarr/training_data/hrrr.zarr"
    
    verify_zarr(zarr_path)