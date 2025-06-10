from .common import *

def destroyer_reality_check_composite(mlcape: np.ndarray, effective_shear: np.ndarray,
                                    srh_01km: np.ndarray, mlcin: np.ndarray,
                                    lcl_height: np.ndarray, pwat: np.ndarray,
                                    pwat_climo: np.ndarray = None) -> np.ndarray:
    """
    Compute Destroyer Reality-Check Composite (DRC)
    
    A "no-hype-allowed" score that only lights up when every critical 
    severe-storm ingredient is simultaneously strong. Any weak link drags 
    the number toward zero, so pretty but hollow soundings stay quiet.
    
    Multiplicative formula exposes weak links:
    DRC = (MLCAPE/2000) × (EffShear/40) × (SRH01/200) × exp(MLCIN/50) 
          × max(0, (2000-LCL)/2000)^0.5 × (PWAT_anom/15)
    
    Args:
        mlcape: Mixed Layer CAPE (J/kg)
        effective_shear: Effective 0-6km wind shear (m/s)
        srh_01km: 0-1km Storm Relative Helicity (m²/s²)
        mlcin: Mixed Layer CIN (J/kg, negative values)
        lcl_height: Lifted Condensation Level height (m)
        pwat: Precipitable water (mm)
        pwat_climo: PWAT climatology (mm) - if None, uses estimate
        
    Returns:
        DRC values (0-1 scale, dimensionless)
    """
    # 1. Instability term: MLCAPE/2000 (clip ≤ 1)
    cape_term = np.minimum(mlcape / 2000.0, 1.0)
    cape_term = np.maximum(cape_term, 0.0)
    
    # 2. Deep-layer shear term: EffShear/40 m/s
    shear_term = np.minimum(effective_shear / 40.0, 1.0)
    shear_term = np.maximum(shear_term, 0.0)
    
    # 3. Low-level helicity term: SRH01/200 m²/s²
    srh_term = np.minimum(srh_01km / 200.0, 1.0)
    srh_term = np.maximum(srh_term, 0.0)
    
    # 4. Capping/CIN term: exp(MLCIN/50) where CIN ≤ 0
    # Strong negative CIN (< -125 J) crushes the score
    # CIN is typically negative, so MLCIN/50 will be negative
    cin_term = np.exp(mlcin / 50.0)  # Exponential decay for strong caps
    cin_term = np.clip(cin_term, 0.0, 1.0)
    
    # 5. Cloud-base height term: max(0, (2000-LCL)/2000)^0.5
    # High LCLs → dusty downdrafts, not photogenic tubes
    lcl_term = np.maximum(0, (2000 - lcl_height) / 2000.0)
    lcl_term = np.power(lcl_term, 0.5)  # Square root to soften penalty
    
    # 6. Moisture depth term: (PWAT - climo)/15 mm (clip 0-1)
    if pwat_climo is not None:
        pwat_anom = pwat - pwat_climo
    else:
        # Use rough CONUS climatology estimate
        pwat_climo_est = 25.0  # mm (seasonal average)
        pwat_anom = pwat - pwat_climo_est
    
    moisture_term = np.clip(pwat_anom / 15.0, 0.0, 1.0)
    
    # Destroyer Reality-Check Composite (multiplicative!)
    # Every factor is 0-1, so one bad ingredient tanks the whole score
    drc = (cape_term * shear_term * srh_term * 
           cin_term * lcl_term * moisture_term)
    
    # Ensure 0-1 bounds (though multiplication should keep it there)
    drc = np.clip(drc, 0.0, 1.0)
    
    # Mask invalid data
    drc = np.where(
        (np.isnan(mlcape)) | (np.isnan(effective_shear)) | (np.isnan(srh_01km)) |
        (np.isnan(mlcin)) | (np.isnan(lcl_height)) | (np.isnan(pwat)),
        np.nan, drc
    )
    
    return drc
