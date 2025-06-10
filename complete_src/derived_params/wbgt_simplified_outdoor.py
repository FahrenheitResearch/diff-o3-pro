from .common import *

def wbgt_simplified_outdoor(wet_bulb_temp: np.ndarray, dry_bulb_temp: np.ndarray, 
                           wind_speed_10m: np.ndarray) -> np.ndarray:
    """
    Simplified outdoor WBGT without time-dependent solar estimation
    
    Uses a constant solar load estimate suitable for daytime conditions.
    More conservative than time-varying estimate.
    
    Args:
        wet_bulb_temp: Wet bulb temperature (°C)
        dry_bulb_temp: Dry bulb (air) temperature (°C)
        wind_speed_10m: 10m wind speed (m/s)
        
    Returns:
        WBGT simplified outdoor values (°C)
    """
    # Fixed solar load for daytime conditions
    solar_boost = 2.5  # °C, conservative daytime solar load
    
    # Estimate black globe with fixed solar load
    black_globe = (dry_bulb_temp + 
                  solar_boost - 
                  (wind_speed_10m * 0.4))
    
    # Black globe cannot be cooler than air temperature
    black_globe = np.maximum(black_globe, dry_bulb_temp)
    
    # WBGT calculation
    wbgt = (0.7 * wet_bulb_temp + 
            0.2 * black_globe + 
            0.1 * dry_bulb_temp)
    
    # Mask invalid data
    wbgt = np.where(
        (np.isnan(wet_bulb_temp)) | (np.isnan(dry_bulb_temp)) | (np.isnan(wind_speed_10m)),
        np.nan, wbgt
    )
    
    return wbgt
