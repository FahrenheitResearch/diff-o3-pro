from .common import *

def craven_brooks_composite(cape: np.ndarray, shear_06km: np.ndarray,
                           srh_03km: np.ndarray) -> np.ndarray:
    """
    Compute Craven-Brooks Composite Parameter
    
    Args:
        cape: CAPE (J/kg)
        shear_06km: 0-6km bulk wind shear (m/s)
        srh_03km: 0-3km storm relative helicity (m²/s²)
        
    Returns:
        Craven-Brooks composite (dimensionless)
    """
    # Normalize components
    cape_norm = cape / 1000.0
    shear_norm = shear_06km / 20.0
    srh_norm = srh_03km / 200.0
    
    # Weight the components
    composite = 0.4 * cape_norm + 0.4 * shear_norm + 0.2 * srh_norm
    
    return np.maximum(composite, 0)
