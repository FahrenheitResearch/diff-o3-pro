from .common import *

def supercell_strength_index(cape: np.ndarray, shear_magnitude: np.ndarray,
                           updraft_helicity: np.ndarray, lcl_height: np.ndarray) -> np.ndarray:
    """
    Compute Supercell Strength Index
    
    Args:
        cape: CAPE (J/kg)
        shear_magnitude: Bulk wind shear magnitude (m/s)
        updraft_helicity: Updraft helicity (m²/s²)
        lcl_height: LCL height (m)
        
    Returns:
        Supercell strength index (dimensionless)
    """
    # Normalized terms
    cape_factor = np.minimum(cape / 2000.0, 2.0)
    shear_factor = np.minimum(shear_magnitude / 30.0, 1.5)
    uh_factor = np.minimum(updraft_helicity / 150.0, 2.0)
    
    # LCL penalty (high LCL reduces strength)
    lcl_factor = np.where(lcl_height > 2500, 0.5,
                         np.where(lcl_height < 1500, 1.0,
                                 1.0 - (lcl_height - 1500) / 2000.0))
    
    strength = cape_factor * shear_factor * uh_factor * lcl_factor
    
    return np.maximum(strength, 0)
