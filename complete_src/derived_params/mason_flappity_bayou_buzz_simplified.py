from .common import *

def mason_flappity_bayou_buzz_simplified(pwat: np.ndarray, mlcape: np.ndarray,
                                       wind_shear_06km: np.ndarray,
                                       u_10m: np.ndarray, v_10m: np.ndarray) -> np.ndarray:
    """
    Simplified MF-Buzz when 850mb fields are not available
    
    Uses surface winds and estimates for missing terms
    
    Args:
        pwat: Precipitable water (mm)
        mlcape: Mixed Layer CAPE (J/kg)
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s)
        u_10m: 10m u-wind component (m/s)
        v_10m: 10m v-wind component (m/s)
        
    Returns:
        MF-Buzz values (0-5 scale)
    """
    # Use reduced version with available fields
    pwat_climo_est = 35.0
    pwat_anom = pwat - pwat_climo_est
    M = np.clip(pwat_anom / 15.0, 0.0, 1.0)
    
    J = np.clip(mlcape / 2500.0, 0.0, 1.0)
    
    S = np.clip(wind_shear_06km / 25.0, 0.0, 1.0)
    
    # Simplified convergence using surface winds
    wind_speed_10m = np.sqrt(u_10m**2 + v_10m**2)
    C_simplified = np.clip(wind_speed_10m / 15.0, 0.0, 1.0)  # Rough proxy
    
    # Simplified vorticity using surface wind shear
    V_simplified = np.clip(wind_shear_06km / 30.0, 0.0, 1.0)  # Another rough proxy
    
    mf_buzz = M + J + S + C_simplified + V_simplified
    mf_buzz = np.clip(mf_buzz, 0.0, 5.0)
    
    # Mask invalid data
    mf_buzz = np.where(
        (np.isnan(pwat)) | (np.isnan(mlcape)) | (np.isnan(wind_shear_06km)) |
        (np.isnan(u_10m)) | (np.isnan(v_10m)),
        np.nan, mf_buzz
    )
    
    return mf_buzz
