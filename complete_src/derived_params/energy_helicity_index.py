from .common import *

def energy_helicity_index(cape: np.ndarray, srh_03km: np.ndarray) -> np.ndarray:
    """
    Compute Energy-Helicity Index (EHI) with anti-saturation damping
    
    EHI = (CAPE/1600) × (SRH/50) × damping_factor
    
    Energy-Helicity Index represents the co-location of instability and low-level 
    rotational potential. Includes damping to prevent "red sea" oversaturation
    in extreme environments while preserving discrimination in moderate cases.
    
    Args:
        cape: CAPE (J/kg) - surface-based or mixed-layer
        srh_03km: 0-3 km Storm Relative Helicity (m²/s²) - preserves sign
        
    Returns:
        EHI values (dimensionless, can be positive or negative)
        
    Interpretation:
        EHI > 1: Notable for supercells
        EHI > 2: Significant tornado potential  
        EHI > 5: Extreme tornado potential
        Positive EHI: Cyclonic (right-moving) supercell potential
        Negative EHI: Anticyclonic (left-moving) supercell potential
    """
    # Basic EHI calculation
    ehi_raw = (cape / 1600.0) * (srh_03km / 50.0)
    
    # Apply damping factor to prevent extreme oversaturation
    # Damping kicks in when |EHI| > 5, scales down exponentially for extreme values
    ehi_abs = np.abs(ehi_raw)
    damping_threshold = 5.0
    
    # Exponential damping: factor = threshold + log(ehi_abs/threshold) for values > threshold
    # This compresses extreme values while preserving moderate ones
    damping_factor = np.where(
        ehi_abs > damping_threshold,
        damping_threshold + np.log(ehi_abs / damping_threshold),
        ehi_abs
    )
    
    # Preserve sign and apply damping
    ehi = np.sign(ehi_raw) * damping_factor
    
    # Quality control - log extreme raw values for monitoring
    extreme_ehi = np.any(np.abs(ehi_raw) > 15)
    if extreme_ehi:
        print(f"🔍 EHI anti-saturation: Raw peaks {np.nanmax(np.abs(ehi_raw)):.1f}, "
              f"damped to {np.nanmax(np.abs(ehi)):.1f}")
    
    # Mask invalid input data (but preserve negative SRH sign)
    ehi = np.where((cape < 0) | (np.isnan(cape)) | (np.isnan(srh_03km)), 
                  np.nan, ehi)
    
    return ehi
