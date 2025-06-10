from .common import *

def kazoo_maxx_composite(sbcape: np.ndarray, shear_06km: np.ndarray, 
                       srh_01km: np.ndarray, pwat: np.ndarray,
                       temp_700: np.ndarray, temp_500: np.ndarray,
                       updraft_helicity: np.ndarray) -> np.ndarray:
    """
    Compute Kazoo's "Everything-to-Eleven" Composite (K-MAX)
    
    A pure hype index that rewards big numbers—the higher, the better—
    no regional penalties, no subtlety. If the atmosphere is flexing 
    anywhere in your HRRR domain, K-MAX will glow.
    
    K_MAX = (SBCAPE / 1000) + (Shear06 / 10) + (SRH01 / 50) + 
            (PWAT_anom / 10) + (LR_700_500 - 6) + (UH_2_5km / 50)
    
    Args:
        sbcape: Surface-based CAPE (J/kg)
        shear_06km: 0-6 km bulk wind shear magnitude (m/s)
        srh_01km: 0-1 km Storm Relative Helicity (m²/s²)
        pwat: Precipitable water (mm)
        temp_700: 700mb temperature (K)
        temp_500: 500mb temperature (K)
        updraft_helicity: 2-5km updraft helicity (m²/s²)
        
    Returns:
        K-MAX values (dimensionless)
    """
    # Convert temperatures to Celsius if in Kelvin
    if np.mean(temp_700) > 200:  # Likely in Kelvin
        temp_700_c = temp_700 - 273.15
    else:
        temp_700_c = temp_700
        
    if np.mean(temp_500) > 200:  # Likely in Kelvin  
        temp_500_c = temp_500 - 273.15
    else:
        temp_500_c = temp_500
    
    # 1. Surface-based CAPE term
    cape_term = sbcape / 1000.0
    
    # 2. 0-6 km bulk shear term
    shear_term = shear_06km / 10.0
    
    # 3. 0-1 km SRH term  
    srh_term = srh_01km / 50.0
    
    # 4. PWAT anomaly term (simplified - use PWAT above climatological value)
    # For CONUS, typical PWAT ranges from 10-40mm, use 25mm as rough average
    pwat_climo = 25.0  # mm (rough CONUS average)
    pwat_anom = (pwat - pwat_climo) / 10.0
    
    # 5. 700-500 mb lapse rate term  
    # Standard atmosphere lapse rate is ~6.5°C/km, use 6°C/km as threshold
    # Height difference: 700mb (~3000m) to 500mb (~5500m) ≈ 2.5km
    lapse_rate = (temp_700_c - temp_500_c) / 2.5  # °C/km
    lapse_term = np.maximum(lapse_rate - 6.0, 0)  # Only positive contributions
    
    # 6. Updraft helicity term
    uh_term = updraft_helicity / 50.0
    
    # K-MAX composite (all additive - no subtractive terms)
    kmax = cape_term + shear_term + srh_term + pwat_anom + lapse_term + uh_term
    
    # Ensure non-negative (K-MAX only climbs)
    kmax = np.maximum(kmax, 0)
    
    # Mask invalid data
    kmax = np.where(
        (np.isnan(sbcape)) | (np.isnan(shear_06km)) | (np.isnan(srh_01km)) | 
        (np.isnan(pwat)) | (np.isnan(temp_700)) | (np.isnan(temp_500)) | 
        (np.isnan(updraft_helicity)), 
        np.nan, kmax
    )
    
    return kmax
