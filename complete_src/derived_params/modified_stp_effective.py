from .common import *

def modified_stp_effective(mlcape: np.ndarray, effective_srh: np.ndarray,
                         effective_shear: np.ndarray, lcl_height: np.ndarray,
                         mlcin: np.ndarray) -> np.ndarray:
    """
    Compute Modified STP using effective layer parameters
    
    Args:
        mlcape: Mixed Layer CAPE (J/kg)
        effective_srh: Effective storm relative helicity (m²/s²)
        effective_shear: Effective bulk wind shear (m/s)
        lcl_height: LCL height (m)
        mlcin: Mixed Layer CIN (J/kg, negative)
        
    Returns:
        Modified STP (dimensionless)
    """
    # Normalize components
    cape_term = np.minimum(mlcape / 1500.0, 2.0)
    
    # LCL term (favorable for low LCLs)
    lcl_term = np.where(lcl_height > 2000, 0,
                       np.where(lcl_height < 1000, 1.0,
                               (2000 - lcl_height) / 1000.0))
    
    # SRH term using effective SRH
    srh_term = effective_srh / 100.0
    
    # Shear term using effective shear
    shear_term = np.where(effective_shear < 10, 0,
                         np.where(effective_shear > 25, 1.5, 
                                 effective_shear / 20.0))
    
    # CIN modification
    cin_factor = np.where(mlcin < -50, 0.5, 1.0)
    
    modified_stp = cape_term * lcl_term * srh_term * shear_term * cin_factor
    
    # Zero out where CAPE is too low
    modified_stp = np.where(mlcape < 100, 0, modified_stp)
    
    return np.maximum(modified_stp, 0)
