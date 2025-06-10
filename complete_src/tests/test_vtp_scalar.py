#!/usr/bin/env python3
"""
Unit tests for VTP scalar values to prevent regression
"""

import numpy as np
import sys
import os

# Add parent directory to path to import derived parameters
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from derived_params.violent_tornado_parameter import violent_tornado_parameter as vtp

def test_vtp_scalar_values():
    """Test VTP with known scalar inputs to ensure proper scaling"""
    
    # Violent tornado environment (EF4+ favorable)
    violent = vtp(
        mlcape=3500,          # High CAPE
        mlcin=-50,            # Minimal CIN
        lcl_height=800,       # Low LCL
        effective_srh=300,    # Strong rotation
        effective_shear=30,   # Strong shear
        cape_03km=150,        # Good low-level buoyancy
        lapse_rate_03km=7.0   # Steep lapse rate
    )
    
    # Moderate environment
    moderate = vtp(
        mlcape=1500,          # Moderate CAPE
        mlcin=-75,            # Some CIN
        lcl_height=900,       # Moderate LCL
        effective_srh=150,    # Moderate rotation
        effective_shear=20,   # Moderate shear
        cape_03km=100,        # Moderate low-level buoyancy
        lapse_rate_03km=6.5   # Standard lapse rate
    )
    
    # Null case (should be very low)
    null_case = vtp(
        mlcape=50,            # Very low CAPE
        mlcin=-25,            # Light CIN
        lcl_height=800,       # Low LCL but no other support
        effective_srh=30,     # Weak rotation
        effective_shear=10,   # Weak shear
        cape_03km=20,         # Low buoyancy
        lapse_rate_03km=5.0   # Weak lapse rate
    )
    
    print(f"Violent environment VTP: {violent}")
    print(f"Moderate environment VTP: {moderate}")
    print(f"Null case VTP: {null_case}")
    
    # Assertions based on expected VTP ranges
    assert violent > 4.0, f"Violent VTP should be > 4.0, got {violent}"
    assert 1.0 < moderate < 2.5, f"Moderate VTP should be 1.0-2.5, got {moderate}"
    assert null_case < 0.2, f"Null case VTP should be < 0.2, got {null_case}"
    
    print("✓ All VTP scalar tests passed!")

if __name__ == "__main__":
    test_vtp_scalar_values()