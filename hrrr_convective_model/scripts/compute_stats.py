import json
import argparse
import numpy as np
import tqdm
import zarr
from pathlib import Path

def main(zarr_path: Path, out: Path):
    """Compute normalization statistics using zarr directly."""
    store = zarr.open(str(zarr_path), mode='r')
    
    # Get data variables (exclude coordinates)
    coord_names = {'time', 'latitude', 'longitude'}
    var_names = [name for name, arr in store.arrays() 
                 if name not in coord_names]
    
    stats = {}
    
    for v in tqdm.tqdm(var_names):
        arr = store[v]
        
        # Sample data to compute stats (avoid loading entire array)
        n_times = arr.shape[0]
        sample_idx = np.linspace(0, n_times-1, min(10, n_times), dtype=int)
        
        # Collect samples
        samples = []
        for idx in sample_idx:
            data = arr[idx].flatten()
            # Remove non-finite values
            data = data[np.isfinite(data)]
            samples.append(data)
        
        # Compute statistics
        all_data = np.concatenate(samples)
        mean = float(np.mean(all_data))
        std = float(np.std(all_data)) + 1e-8  # Avoid division by zero
        
        stats[v] = {
            "mean": mean,
            "std": std
        }
    
    # Ensure output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ wrote stats for {len(stats)} variables to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, type=Path, dest="zarr_path")
    parser.add_argument("--out", default="data/stats.json", type=Path)
    args = parser.parse_args()
    main(args.zarr_path, args.out)