#!/usr/bin/env python
"""
Convert a directory tree of HRRR GRIB2 files to period‑chunked Zarr cubes.
Uses cfgrib → xarray → Zarr; runs in parallel with dask.
"""
from pathlib import Path
import argparse, json, xarray as xr, dask
import cfgrib, zarr, numcodecs, tqdm, warnings
from dask.distributed import Client
warnings.filterwarnings("ignore", category=RuntimeWarning)

VARS = {    # edit to taste
    "REFC": {},                             # composite refl
    "REFD": {"typeOfLevel": "heightAboveGround", "level": 1000},
    "CAPE": {"typeOfLevel": "atmosphereLayer", "bottomLevel":255,"topLevel":0},
    "CIN":  {"typeOfLevel": "atmosphereLayer", "bottomLevel":255,"topLevel":0},
    "ACPCP":{},
    "TMP":  {"typeOfLevel":"heightAboveGround","level":2},
    "DPT":  {"typeOfLevel":"heightAboveGround","level":2},
    "UGRD":{"typeOfLevel":"heightAboveGround","level":10},
    "VGRD":{"typeOfLevel":"heightAboveGround","level":10},
}

def open_single(fp: Path):
    """Open the subset of variables in one GRIB2 file as xarray Dataset."""
    datasets = []
    for k, flt in VARS.items():
        try:
            ds = xr.open_dataset(fp, engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": k, **flt}))
            # Ensure time coordinate exists
            if 'time' not in ds.coords and 'valid_time' in ds.coords:
                ds = ds.rename({'valid_time': 'time'})
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {k} from {fp.name}: {e}")
            continue
    
    if not datasets:
        return None
    
    # Merge all datasets
    merged = xr.merge(datasets)
    
    # Ensure we have a time coordinate
    if 'time' not in merged.coords:
        # Try to extract time from filename or use step/valid_time
        if 'step' in merged.coords:
            merged = merged.rename({'step': 'time'})
        elif 'valid_time' in merged.coords:
            merged = merged.rename({'valid_time': 'time'})
    
    return merged

def main(src_dir: Path, out_dir: Path, chunks: str = "time=1,lat=512,lon=512"):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.grib2"))
    
    if not files:
        raise ValueError(f"No GRIB2 files found in {src_dir}")
    
    store = zarr.DirectoryStore(out_dir / "hrrr.zarr")
    consolidated = []
    
    # Use Client context manager properly
    with Client(n_workers=4, threads_per_worker=2, memory_limit='4GB') as client:
        print(f"Processing {len(files)} GRIB2 files...")
        
        for fp in tqdm.tqdm(files):
            ds = open_single(fp)
            if ds is not None and len(ds.data_vars) > 0:
                # Ensure time coordinate exists
                if 'time' in ds.coords:
                    consolidated.append(ds)
                else:
                    print(f"Skipping {fp.name}: no time coordinate")
            else:
                print(f"Skipping {fp.name}: no valid data")
        
        if not consolidated:
            raise ValueError("No valid datasets to process! Check your GRIB2 files.")
        
        print(f"Concatenating {len(consolidated)} valid datasets...")
        full = xr.concat(consolidated, dim="time").sortby("time")
        
        # Apply chunking
        full = full.chunk(xr.core.utils.parse_chunks(chunks))
        
        print("Writing to Zarr...")
        full.to_zarr(store, mode="w", consolidated=True)
    
    # Write manifest
    meta = {
        "variables": list(full.data_vars),
        "shape": dict(full.sizes),
        "time_range": {
            "start": str(full.time.min().values),
            "end": str(full.time.max().values),
            "steps": len(full.time)
        }
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ Wrote {store} with {len(full.time)} timesteps")
    print(f"   Variables: {list(full.data_vars)}")
    print(f"   Time range: {meta['time_range']['start']} to {meta['time_range']['end']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="path to directory of GRIB2 files")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    args = p.parse_args()
    main(args.src, args.out)