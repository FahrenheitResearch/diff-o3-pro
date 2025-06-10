from .common import *

def wbgt_shade(wet_bulb_temp: np.ndarray, dry_bulb_temp: np.ndarray) -> np.ndarray:
    """
    Compute Wet Bulb Globe Temperature for shaded conditions (no solar load)
    
    WBGT_shade = 0.7 * WB + 0.3 * DB
    
    Used for indoor work environments, shaded areas, and nighttime conditions.
    
    Args:
        wet_bulb_temp: Wet bulb temperature (°C)
        dry_bulb_temp: Dry bulb (air) temperature (°C)
        
    Returns:
        WBGT shade values (°C)
        
    Heat stress thresholds:
        > 32°C: Stop activity
        28-32°C: Extreme caution 
        25-28°C: High caution
        < 25°C: Generally safe
    """
    # Input validation
    if wet_bulb_temp is None or dry_bulb_temp is None:
        raise ValueError("Temperature inputs cannot be None")
    
    # WBGT calculation for shaded conditions
    wbgt = 0.7 * wet_bulb_temp + 0.3 * dry_bulb_temp
    
    # Mask invalid data
    wbgt = np.where(
        (np.isnan(wet_bulb_temp)) | (np.isnan(dry_bulb_temp)),
        np.nan, wbgt
    )
    
    return wbgt
