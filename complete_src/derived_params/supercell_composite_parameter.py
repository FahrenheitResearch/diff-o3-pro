from .common import *

def supercell_composite_parameter(mucape: np.ndarray, effective_srh: np.ndarray, 
                                effective_shear: np.ndarray, mucin: np.ndarray = None) -> np.ndarray:
    """
    Compute Supercell Composite Parameter (SCP) - Official SPC Recipe
    
    SCP = (muCAPE / 1000) × (ESRH / 50) × (EBWD / 20) × (-40 / muCIN)
    
    Uses the official three-ingredient (+ CIN penalty) recipe exactly as SPC.
    Values > 10 indicate extreme 'overlap'; operational guidance often focuses on 1 – 10.
    
    Ingredients:
    - muCAPE: most-unstable CAPE (J kg⁻¹)
    - ESRH: effective 0-3 km storm-relative helicity (m² s⁻²) 
    - EBWD: effective bulk-wind difference (m s⁻¹)
    - muCIN: most-unstable CIN (J kg⁻¹, negative)
    
    Args:
        mucape: Most-Unstable CAPE (J/kg) - HRRR field MUCAPE
        effective_srh: Effective Storm Relative Helicity (m²/s²) - HRRR field ESRHL
        effective_shear: Effective Bulk Wind Difference (m/s) - derived parameter
        mucin: Most-Unstable CIN (J/kg, negative values) - HRRR field MUCIN [optional] 
        
    Returns:
        Tuple of (scp_raw, scp_plot):
        - scp_raw: Untouched SCP field (≥ 0, no upper limit)
        - scp_plot: Clipped for mapping (-2 to 10, matches SPC graphics)
        
    References:
        Thompson et al. (2012), Gropp & Davenport (2018)
    """
    
    # ========================================================================
    # MUCIN SIGN FIX - Ensure HRRR MUCIN field is negative for SPC formula
    # ========================================================================
    if mucin is not None:
        # Force MUCIN to negative values (HRRR may store as positive magnitude)
        mucin = -np.abs(mucin)
    
    # ========================================================================
    # QUALITY FLAGS - Log outliers for debugging
    # ========================================================================
    extreme_cape = np.any(mucape > 6000)
    extreme_srh = np.any(effective_srh > 800)
    extreme_shear = np.any(effective_shear > 60)
    
    if extreme_cape or extreme_srh or extreme_shear:
        print(f"🔍 SCP outliers detected: muCAPE>{6000 if extreme_cape else 'OK'}, "
              f"SRH>{800 if extreme_srh else 'OK'}, Shear>{60 if extreme_shear else 'OK'}")
    
    # ========================================================================
    # 1. CAPE TERM - muCAPE ÷ 1000
    # ========================================================================
    cape_term = mucape / 1000.0
    
    # ========================================================================
    # 2. SRH TERM - SRH ÷ 50 (force negative values to 0 before dividing)
    # ========================================================================
    srh_positive = np.maximum(effective_srh, 0.0)  # Force negatives to 0 first
    srh_term = srh_positive / 50.0
    
    # ========================================================================
    # 3. SHEAR TERM - SPC-compliant piece-wise EBWD scaling
    # ========================================================================
    # 0 when EBWD < 10 m s⁻¹
    # linear from 0→1 between 10–20 m s⁻¹: (EBWD-10)/10
    # 1 once EBWD ≥ 20 m s⁻¹
    shear_term = np.clip((effective_shear - 10.0) / 10.0, 0.0, 1.0)
    
    # ========================================================================
    # 4. CIN WEIGHT - Official SPC formula (two-part gate)
    # ========================================================================
    # Official SPC Rule:
    # If inhibition is weak (muCIN > -40 J/kg), the penalty is removed (term = 1.0)
    # Otherwise, the penalty is proportional: -40 / muCIN
    if mucin is not None:
        cin_weight = np.where(
            mucin > -40.0,            # Weak inhibition
            1.0,                      # No penalty
            -40.0 / mucin             # Proportional penalty
        )
        # Ensure valid range (should be 0.0 to 1.0 naturally)
        cin_weight = np.clip(cin_weight, 0.0, 1.0)
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
    
    # Mask low-CAPE areas to avoid blue speckles (optional polish)
    low_cape_mask = mucape < 50.0  # J/kg threshold
    scp_raw = np.where(low_cape_mask, 0.0, scp_raw)
    
    # Store untouched field, then create clipped version for mapping
    scp_plot = np.clip(scp_raw, -2, 10)  # SPC-compliant range per spec §2
    
    return scp_raw, scp_plot


def supercell_composite_parameter_legacy(mucape: np.ndarray, srh_03km: np.ndarray, 
                                        shear_06km: np.ndarray, mlcin: np.ndarray = None) -> np.ndarray:
    """
    Legacy SCP implementation - kept for backwards compatibility
    
    Uses 0-3km SRH instead of ESRH and ML-CIN instead of MU-CIN.
    For operational use, prefer the main supercell_composite_parameter function.
    """
    # CAPE term (only positive CAPE contributes)
    cape_term = np.maximum(mucape / 1000.0, 0)
    
    # SRH term - PRESERVE SIGN! Negative SRH = left-moving storms
    # Do NOT take absolute value or force to positive
    srh_term = srh_03km / 50.0
    
    # Shear term with SPC ops cap (stops background carpets)
    # Cap at 1.0 once shear >= 20 m/s (what SPC does in operations)  
    shear_term = np.minimum(shear_06km / 20.0, 1.0)
    
    # SCP calculation - can be positive or negative
    scp = cape_term * srh_term * shear_term
    
    # Apply CIN gate to knock out carpets (optional but recommended)
    if mlcin is not None:
        scp = scp * cin_gate(mlcin)
    
    # Mask invalid input data (but allow negative SRH)
    scp = np.where((mucape < 0) | (np.isnan(mucape)) | 
                  (np.isnan(srh_03km)) | (shear_06km < 0) | (np.isnan(shear_06km)), 
                  np.nan, scp)
    
    return scp