from .common import *

def supercell_composite_parameter_effective(mucape: np.ndarray, 
                                           effective_srh: np.ndarray,
                                           effective_shear: np.ndarray,
                                           mucin: np.ndarray = None) -> np.ndarray:
    """
    Compute SCP using effective layers - Official SPC Recipe
    
    SCP = (muCAPE / 1000) × (ESRH / 50) × (EBWD / 20) × (-40 / muCIN)
    
    Uses effective layers with the official SPC normalization and CIN penalty.
    
    Args:
        mucape: Most-Unstable CAPE (J/kg)
        mucin: Most-Unstable CIN (J/kg, negative values)
        effective_srh: Effective SRH (m²/s²)
        effective_shear: Effective bulk shear (m/s)
        
    Returns:
        Tuple of (scp_raw, scp_plot) following official SPC recipe
    """
    
    # ========================================================================
    # QUALITY FLAGS - Log outliers for debugging
    # ========================================================================
    extreme_cape = np.any(mucape > 6000)
    extreme_srh = np.any(effective_srh > 800) 
    extreme_shear = np.any(effective_shear > 60)
    
    if extreme_cape or extreme_srh or extreme_shear:
        print(f"🔍 SCP-Effective outliers: muCAPE>{6000 if extreme_cape else 'OK'}, "
              f"ESRH>{800 if extreme_srh else 'OK'}, EBWD>{60 if extreme_shear else 'OK'}")
    
    # ========================================================================
    # 1. CAPE TERM - muCAPE ÷ 1000
    # ========================================================================
    cape_term = mucape / 1000.0
    
    # ========================================================================
    # 2. SRH TERM - ESRH ÷ 50 (force negative values to 0 before dividing)
    # ========================================================================
    esrh_positive = np.maximum(effective_srh, 0.0)  # Force negatives to 0 first
    srh_term = esrh_positive / 50.0
    
    # ========================================================================
    # 3. SHEAR TERM - EBWD with official SPC scaling
    # ========================================================================
    shear_term = np.where(
        effective_shear < 10.0, 0.0,                      # 0 when < 10 m/s
        np.where(effective_shear < 20.0, effective_shear / 20.0,  # Linear 10-20 m/s  
                 1.0)                                      # 1 when >= 20 m/s
    )
    
    # ========================================================================
    # 4. CIN WEIGHT - Official SPC formula (optional)
    # ========================================================================
    if mucin is not None:
        cin_weight = np.where(
            mucin >= -40.0,           # Weak cap
            1.0,                      # No penalty
            (-40.0) / mucin           # -40 ÷ muCIN
        )
        cin_weight = np.clip(cin_weight, 0.0, 1.0)  # Floor at 0, cap at 1
    else:
        cin_weight = 1.0  # No CIN penalty if not provided
    
    # ========================================================================
    # 5. FINAL SCP - CAPE × SRH × Shear × CIN_weight
    # ========================================================================
    scp_raw = cape_term * srh_term * shear_term * cin_weight
    
    # ========================================================================
    # 6. POSITIVE-ONLY OUTPUT - Set negative/NaN to 0
    # ========================================================================
    # Mask invalid input data first
    valid_data = (
        np.isfinite(mucape) & (mucape >= 0) &
        np.isfinite(effective_srh) &
        np.isfinite(effective_shear) & (effective_shear >= 0)
    )
    
    # Add mucin validation only if provided
    if mucin is not None:
        valid_data = valid_data & np.isfinite(mucin)
    
    # After the product, set any negative or NaN to 0
    scp_raw = np.where(valid_data & np.isfinite(scp_raw) & (scp_raw >= 0), scp_raw, 0.0)
    
    # Store untouched field, then create clipped version for mapping
    scp_plot = np.clip(scp_raw, -2, 10)  # Matches SPC graphics
    
    return scp_raw, scp_plot
