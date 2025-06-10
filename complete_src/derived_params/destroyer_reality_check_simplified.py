from .common import *

def destroyer_reality_check_simplified(mlcape: np.ndarray, wind_shear_06km: np.ndarray,
                                     srh_01km: np.ndarray, mlcin: np.ndarray,
                                     lcl_height: np.ndarray, pwat: np.ndarray) -> np.ndarray:
    """
    Simplified Destroyer Reality-Check using available HRRR fields
    
    Uses bulk 0-6km shear as proxy for effective shear
    
    Args:
        mlcape: Mixed Layer CAPE (J/kg)
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s)
        srh_01km: 0-1km Storm Relative Helicity (m²/s²)
        mlcin: Mixed Layer CIN (J/kg, negative)
        lcl_height: LCL height (m)
        pwat: Precipitable water (mm)
        
    Returns:
        DRC values (0-1 scale)
    """
    return destroyer_reality_check_composite(
        mlcape, wind_shear_06km, srh_01km, mlcin, lcl_height, pwat
    )
