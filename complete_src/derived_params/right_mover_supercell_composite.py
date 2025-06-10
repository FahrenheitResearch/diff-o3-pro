from .common import *

def right_mover_supercell_composite(mucape: np.ndarray, shear_06km: np.ndarray,
                                  srh_03km: np.ndarray, storm_motion_u: float = 10.0,
                                  storm_motion_v: float = 5.0) -> np.ndarray:
    """
    Compute Right-Moving Supercell Composite Parameter
    
    Args:
        mucape: Most Unstable CAPE (J/kg)
        shear_06km: 0-6km bulk wind shear (m/s)
        srh_03km: 0-3km storm relative helicity (m²/s²)
        storm_motion_u: Storm motion U component (m/s)
        storm_motion_v: Storm motion V component (m/s)
        
    Returns:
        Right-mover supercell composite (dimensionless)
    """
    # Enhanced weighting for right-moving storms
    cape_term = np.minimum(mucape / 1200.0, 2.5)
    shear_term = np.where(shear_06km < 15, 0, shear_06km / 25.0)
    srh_term = np.maximum(srh_03km / 75.0, 0)
    
    # Right-mover enhancement
    storm_speed = np.sqrt(storm_motion_u**2 + storm_motion_v**2)
    motion_factor = np.minimum(storm_speed / 15.0, 1.5)
    
    composite = cape_term * shear_term * srh_term * motion_factor
    
    return np.maximum(composite, 0)
