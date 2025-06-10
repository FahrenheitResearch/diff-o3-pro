from .common import *

def seqouigrove_weird_west_composite(pwat: np.ndarray, sbcape: np.ndarray, 
                                   temp_500: np.ndarray, vorticity_500: np.ndarray,
                                   lcl_height: np.ndarray, 
                                   pwat_climo: np.ndarray = None,
                                   temp_500_climo: np.ndarray = None) -> np.ndarray:
    """
    Compute Seqouigrove Weird-West Composite (SW²C)
    
    Highlights weird moisture-charged events in normally dry western US regions
    
    Args:
        pwat: Precipitable water (mm)
        sbcape: Surface-based CAPE (J/kg)
        temp_500: 500mb temperature (K)
        vorticity_500: 500mb absolute vorticity (s⁻¹)
        lcl_height: Lifted condensation level height (m)
        pwat_climo: PWAT climatology (mm) - if None, uses simple estimate
        temp_500_climo: 500mb temp climatology (K) - if None, uses simple estimate
        
    Returns:
        SW²C values (dimensionless)
    """
    # Moisture pop: PWAT anomaly
    if pwat_climo is not None:
        pwat_anom = pwat - pwat_climo
    else:
        # Simple western US climatology estimate (typically 5-15mm)
        west_climo = 12.0  # mm (reasonable for Sierra/Great Basin)
        pwat_anom = pwat - west_climo
    
    # Desert juice: Surface CAPE normalized by 250 J/kg
    # (rewards rare southwestern CAPE spikes)
    cape_term = sbcape / 250.0
    
    # Cold-core kick: 500mb temperature anomaly × 0.1
    if temp_500_climo is not None:
        temp_anom = (temp_500 - temp_500_climo) * 0.1
    else:
        # Rough summer 500mb temp for western US (~265K)
        temp_climo = 265.0  # K
        temp_anom = (temp_500 - temp_climo) * 0.1
    
    # Vorticity sauce: 500mb absolute vorticity × 0.05
    # Take absolute value to capture both cyclonic maxima
    vort_term = np.abs(vorticity_500) * 0.05
    
    # Dry-air tax: LCL height penalty (divide by 1000m)
    # High LCL = dry air = penalty for desert regions
    lcl_penalty = lcl_height / 1000.0
    
    # Seqouigrove Weird-West Composite
    sw2c = pwat_anom + cape_term + temp_anom + vort_term - lcl_penalty
    
    # Mask invalid data
    sw2c = np.where(
        (np.isnan(pwat)) | (np.isnan(sbcape)) | (np.isnan(temp_500)) | 
        (np.isnan(vorticity_500)) | (np.isnan(lcl_height)), 
        np.nan, sw2c
    )
    
    return sw2c
