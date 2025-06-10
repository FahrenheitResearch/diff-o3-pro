from .common import *

def seqouigrove_weird_west_simplified(pwat: np.ndarray, sbcape: np.ndarray,
                                    temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                                    u_500: np.ndarray, v_500: np.ndarray,
                                    temp_500: np.ndarray) -> np.ndarray:
    """
    Simplified SW²C using available HRRR fields without climatology
    
    Args:
        pwat: Precipitable water (mm)
        sbcape: Surface-based CAPE (J/kg)
        temp_2m: 2m temperature (°C)
        dewpoint_2m: 2m dewpoint (°C)
        u_500: 500mb U wind (m/s)
        v_500: 500mb V wind (m/s)
        temp_500: 500mb temperature (K)
        
    Returns:
        SW²C values (dimensionless)
    """
    # Estimate LCL height
    lcl_height = crude_lcl_estimate(temp_2m, dewpoint_2m)
    
    # Estimate wind speed for vorticity calculation
    wind_speed_500 = np.sqrt(u_500**2 + v_500**2)
    
    # Use a representative latitude for western US (37°N - roughly Reno/SF area)
    representative_lat = np.full_like(pwat, 37.0)
    abs_vort = absolute_vorticity_500_estimate(wind_speed_500, representative_lat)
    
    # Calculate SW²C
    return seqouigrove_weird_west_composite(
        pwat, sbcape, temp_500, abs_vort, lcl_height
    )
