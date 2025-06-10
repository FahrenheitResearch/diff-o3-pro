#!/usr/bin/env python3
"""
HRRR Derived Parameters Module
Computes composite severe weather indices from available HRRR fields
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional, Tuple
import warnings

# MetPy integration for advanced meteorological calculations
try:
    from metpy.calc import (
        heat_index, wind_chill, mixing_ratio, wind_speed, wind_direction,
        potential_temperature, equivalent_potential_temperature, 
        wet_bulb_temperature, relative_humidity_from_mixing_ratio,
        saturation_mixing_ratio, brunt_vaisala_frequency,
        richardson_number_bulk, gradient_richardson_number
    )
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    print("MetPy not available - some advanced derived parameters will not be computed")
    METPY_AVAILABLE = False


def cin_gate(cin: np.ndarray, hi: float = -50.0, lo: float = -100.0) -> np.ndarray:
    """
    Return 0-1 weight: 1 when CIN >= hi, 0 when CIN <= lo.
    
    Global helper for knocking out carpets where cap is too strong.
    Use in SCP/EHI and other composites that need CIN penalty.
    
    Args:
        cin: CIN values (J/kg, negative)
        hi: Upper threshold (weaker cap) - full weight
        lo: Lower threshold (stronger cap) - zero weight
        
    Returns:
        Weight array 0.0-1.0
    """
    return np.clip((cin - lo) / (hi - lo), 0.0, 1.0)

