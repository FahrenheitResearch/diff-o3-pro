from .common import *

def significant_tornado_parameter_fixed(sbcape: np.ndarray, srh_01km: np.ndarray,
                                      wind_shear_06km: np.ndarray, lcl_height: np.ndarray) -> np.ndarray:
    """
    Compute STP Fixed Layer Version (Thompson et al. 2004)
    
    STP Fixed = (SBCAPE/1500) × (0-1km SRH/150) × (0-6km shear/12) × ((2000-LCL)/1000)
    
    This is the classic fixed-layer version without CIN term.
    Uses surface-based CAPE and fixed 0-1km SRH, 0-6km shear layers.
    
    Args:
        sbcape: Surface-Based CAPE (J/kg)
        srh_01km: 0-1 km Storm Relative Helicity (m²/s²)
        wind_shear_06km: 0-6 km bulk wind shear magnitude (m/s)
        lcl_height: LCL height (m AGL)
        
    Returns:
        STP Fixed values (dimensionless, always ≥ 0)
    """
    # 1. CAPE term: SBCAPE/1500
    cape_term = np.maximum(sbcape / 1500.0, 0)
    
    # 2. SRH term: 0-1km SRH/150 (only positive values)
    srh_term = np.maximum(srh_01km / 150.0, 0)
    
    # 3. Shear term: 0-6km shear/12 m/s
    shear_term = np.maximum(wind_shear_06km / 12.0, 0)
    
    # 4. LCL term: (2000-LCL)/1000 with clipping
    # LCL < 1000m → 1.0, LCL > 2000m → 0.0
    lcl_term = np.where(lcl_height < 1000, 1.0,
                       np.where(lcl_height > 2000, 0.0,
                               (2000 - lcl_height) / 1000.0))
    
    # STP Fixed calculation
    stp_fixed = cape_term * srh_term * shear_term * lcl_term
    
    # Zero out where CAPE is too low for convection
    stp_fixed = np.where(sbcape < 100, 0, stp_fixed)
    
    # Ensure non-negative
    stp_fixed = np.maximum(stp_fixed, 0.0)
    
    # Mask invalid input data
    stp_fixed = np.where((sbcape < 0) | (np.isnan(sbcape)) | 
                        (np.isnan(srh_01km)) | (np.isnan(wind_shear_06km)) |
                        (np.isnan(lcl_height)), np.nan, stp_fixed)
    
    return stp_fixed
