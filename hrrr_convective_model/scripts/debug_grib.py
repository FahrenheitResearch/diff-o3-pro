#!/usr/bin/env python
"""Debug script to see what's actually in HRRR GRIB files."""
import xarray as xr
import sys

if len(sys.argv) > 1:
    grib_file = sys.argv[1]
else:
    grib_file = "data/raw/hrrr.20250610.t00z.wrfprsf00.grib2"

print(f"Inspecting {grib_file}\n")

# Try to open without filters to see all variables
try:
    print("Opening entire file (may take a moment)...")
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    print(f"\nFound {len(ds.data_vars)} variables:")
    for var in list(ds.data_vars)[:10]:  # Show first 10
        print(f"  - {var}")
    if len(ds.data_vars) > 10:
        print(f"  ... and {len(ds.data_vars) - 10} more")
except Exception as e:
    print(f"Could not open entire file: {e}")

# Try specific variables
print("\n" + "="*50 + "\n")

test_vars = [
    ("REFC", {}),
    ("refc", {}),
    ("CAPE", {"typeOfLevel": "atmosphereLayer"}),
    ("cape", {"typeOfLevel": "atmosphereLayer"}),
    ("TMP", {"typeOfLevel": "heightAboveGround", "level": 2}),
    ("t2m", {}),
    ("2t", {}),
]

for var_name, filters in test_vars:
    try:
        filter_keys = {"shortName": var_name}
        filter_keys.update(filters)
        
        ds = xr.open_dataset(
            grib_file,
            engine="cfgrib", 
            backend_kwargs={"filter_by_keys": filter_keys}
        )
        
        print(f"\n✓ {var_name} with filters {filters}:")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Coords: {list(ds.coords)}")
        print(f"  Dims: {list(ds.dims)}")
        
        # Show first variable details
        if ds.data_vars:
            first_var = list(ds.data_vars)[0]
            print(f"  Shape of {first_var}: {ds[first_var].shape}")
            
    except Exception as e:
        print(f"\n✗ {var_name}: {str(e)[:100]}")

# Try with paramId
print("\n" + "="*50 + "\n")
print("Trying with paramId:")

param_tests = [
    (260257, "REFC"),  # Composite reflectivity
    (59, "CAPE"),      # CAPE  
    (167, "2t"),       # 2m temperature
    (165, "10u"),      # 10m u-wind
]

for param_id, desc in param_tests:
    try:
        ds = xr.open_dataset(
            grib_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"paramId": param_id}}
        )
        print(f"\n✓ paramId {param_id} ({desc}):")
        print(f"  Variables: {list(ds.data_vars)}")
        if ds.data_vars:
            first_var = list(ds.data_vars)[0]
            print(f"  Shape: {ds[first_var].shape}")
    except Exception as e:
        print(f"\n✗ paramId {param_id} ({desc}): {str(e)[:50]}")