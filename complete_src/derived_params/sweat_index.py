from .common import *

def sweat_index(temp_850: np.ndarray, temp_500: np.ndarray, 
               dewpoint_850: np.ndarray, u_850: np.ndarray, v_850: np.ndarray,
               u_500: np.ndarray, v_500: np.ndarray) -> np.ndarray:
    """
    Compute SWEAT (Severe Weather Threat) Index
    
    Args:
        temp_850: 850mb temperature (°C)
        temp_500: 500mb temperature (°C)
        dewpoint_850: 850mb dewpoint (°C)
        u_850: 850mb U wind (m/s)
        v_850: 850mb V wind (m/s)
        u_500: 500mb U wind (m/s)
        v_500: 500mb V wind (m/s)
        
    Returns:
        SWEAT index (dimensionless)
    """
    # Calculate wind speeds
    wspd_850 = np.sqrt(u_850**2 + v_850**2)
    wspd_500 = np.sqrt(u_500**2 + v_500**2)
    
    # Calculate wind directions
    wdir_850 = np.degrees(np.arctan2(-u_850, -v_850)) % 360
    wdir_500 = np.degrees(np.arctan2(-u_500, -v_500)) % 360
    
    # Total Totals term
    tt = temp_850 + dewpoint_850 - 2 * temp_500
    
    # Wind speed terms
    ws850_term = np.where(wspd_850 > 15, 12.5 * (wspd_850 - 15), 0)
    ws500_term = np.where(wspd_500 > 15, 2 * (wspd_500 - 15), 0)
    
    # Wind direction term (only if winds from SW quadrant)
    wd_diff = (wdir_500 - wdir_850) % 360
    wd_term = np.where(
        (wdir_850 >= 130) & (wdir_850 <= 250) & 
        (wdir_500 >= 210) & (wdir_500 <= 310) &
        (wd_diff > 0) & (wspd_850 >= 15) & (wspd_500 >= 15),
        125 * (np.sin(np.radians(wd_diff)) + 0.2),
        0
    )
    
    sweat = 12 * tt + 20 * np.maximum(tt - 49, 0) + ws850_term + ws500_term + wd_term
    
    return np.maximum(sweat, 0)
