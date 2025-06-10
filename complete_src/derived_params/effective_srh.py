from .common import *

def effective_srh(srh_data: np.ndarray, cape: np.ndarray, cin: np.ndarray,
                 lcl_height: np.ndarray) -> np.ndarray:
    """
    Compute Effective Storm Relative Helicity (simplified version)
    
    Args:
        srh_data: Storm relative helicity (m²/s²)
        cape: CAPE (J/kg)
        cin: CIN (J/kg, negative values)
        lcl_height: LCL height (m)
        
    Returns:
        Effective SRH (m²/s²)
    """
    # Effective layer criteria (simplified)
    effective_mask = (
        (cape >= 100) &           # Minimum CAPE
        (cin >= -250) &           # Not too much CIN  
        (lcl_height <= 2500)      # Reasonable LCL height
    )
    
    return np.where(effective_mask, srh_data, 0)
