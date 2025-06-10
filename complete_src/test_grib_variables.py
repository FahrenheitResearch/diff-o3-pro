#!/usr/bin/env python3
"""
Test what GRIB variables are actually loadable with cfgrib
"""

import cfgrib
import sys
from pathlib import Path

def test_variable_loading(grib_file):
    """Test loading specific variables to see what works"""
    
    # Test reflectivity
    print("Testing reflectivity variables...")
    try:
        ds = cfgrib.open_dataset(grib_file, 
                               filter_by_keys={'shortName': 'refc'},
                               backend_kwargs={'indexpath': ''})
        print("✅ REFC (composite reflectivity) loaded:", list(ds.data_vars.keys()))
    except Exception as e:
        print("❌ REFC failed:", str(e))
    
    try:
        ds = cfgrib.open_dataset(grib_file,
                               filter_by_keys={'shortName': 'refd', 'level': 1000},
                               backend_kwargs={'indexpath': ''})
        print("✅ REFD 1km loaded:", list(ds.data_vars.keys()))
    except Exception as e:
        print("❌ REFD 1km failed:", str(e))
    
    # Test CAPE
    print("\nTesting CAPE variables...")
    try:
        ds = cfgrib.open_dataset(grib_file,
                               filter_by_keys={'shortName': 'cape', 'typeOfLevel': 'surface'},
                               backend_kwargs={'indexpath': ''})
        print("✅ Surface CAPE loaded:", list(ds.data_vars.keys()))
    except Exception as e:
        print("❌ Surface CAPE failed:", str(e))
        
    try:
        ds = cfgrib.open_dataset(grib_file,
                               filter_by_keys={'shortName': 'cape', 'typeOfLevel': 'pressureFromGroundLayer'},
                               backend_kwargs={'indexpath': ''})
        print("✅ Mixed layer CAPE loaded:", list(ds.data_vars.keys()))
    except Exception as e:
        print("❌ Mixed layer CAPE failed:", str(e))
    
    # Test dewpoint
    print("\nTesting dewpoint variables...")
    try:
        ds = cfgrib.open_dataset(grib_file,
                               filter_by_keys={'shortName': 'dpt', 'level': 850},
                               backend_kwargs={'indexpath': ''})
        print("✅ DPT 850mb loaded:", list(ds.data_vars.keys()))
    except Exception as e:
        print("❌ DPT 850mb failed:", str(e))

if __name__ == "__main__":
    grib_file = "./test_april/hrrr.t00z.wrfprsf00.grib2"
    test_variable_loading(grib_file)