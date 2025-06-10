#!/usr/bin/env python3
"""
Unit tests for Violent Tornado Parameter (VTP)
Tests against Hampshire et al. 2018 reference profiles
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derived_params.violent_tornado_parameter import violent_tornado_parameter


def test_vtp_reference_profiles():
    """Test VTP with reference profiles from Hampshire et al. 2018"""
    
    # Reference profiles as specified
    # Scenario    MLCAPE  ML-CIN  MLLCL   ESRH  EBWD   CAPE 0-3km  LR 0-3km
    # null        50      -25     800m    30    10m/s  20         5.0°C/km  
    # moderate    1500    -75     900m    150   20m/s  100        6.5
    # violent     3500    -50     800m    300   30m/s  150        7.0
    
    # Null environment - should produce very low VTP
    vtp_null = violent_tornado_parameter(
        mlcape=np.array([50]),
        mlcin=np.array([-25]),
        lcl_height=np.array([800]),
        effective_srh=np.array([30]),
        effective_shear=np.array([10]),
        cape_03km=np.array([20]),
        lapse_rate_03km=np.array([5.0])
    )
    
    # Moderate environment - should produce VTP 1.0-2.5  
    vtp_mod = violent_tornado_parameter(
        mlcape=np.array([1500]),
        mlcin=np.array([-75]),
        lcl_height=np.array([900]),
        effective_srh=np.array([150]),
        effective_shear=np.array([20]),
        cape_03km=np.array([100]),
        lapse_rate_03km=np.array([6.5])
    )
    
    # Violent environment - should produce VTP > 4.0
    vtp_violent = violent_tornado_parameter(
        mlcape=np.array([3500]),
        mlcin=np.array([-50]),
        lcl_height=np.array([800]),
        effective_srh=np.array([300]),
        effective_shear=np.array([30]),
        cape_03km=np.array([150]),
        lapse_rate_03km=np.array([7.0])
    )
    
    # Acceptance criteria assertions
    assert vtp_null.max() < 0.1, f"Null VTP too high: {vtp_null.max():.3f}"
    assert 1.0 < vtp_mod.max() < 2.5, f"Moderate VTP out of range: {vtp_mod.max():.3f}"
    assert vtp_violent.max() > 4.0, f"Violent VTP too low: {vtp_violent.max():.3f}"
    
    print(f"✅ VTP Unit Tests Passed:")
    print(f"  Null environment: VTP = {vtp_null[0]:.3f} (< 0.1)")
    print(f"  Moderate environment: VTP = {vtp_mod[0]:.3f} (1.0-2.5)")
    print(f"  Violent environment: VTP = {vtp_violent[0]:.3f} (> 4.0)")


if __name__ == "__main__":
    test_vtp_reference_profiles()
    print("All VTP tests passed! ✅")