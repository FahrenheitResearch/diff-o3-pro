from .common import *

def wbgt_estimated_outdoor(wet_bulb_temp: np.ndarray, dry_bulb_temp: np.ndarray, 
                          wind_speed_10m: np.ndarray) -> np.ndarray:
    """
    Compute Wet Bulb Globe Temperature with estimated solar load for outdoor conditions
    
    WBGT = 0.7 * WB + 0.2 * BG + 0.1 * DB
    where BG (Black Globe) is estimated from solar load and wind cooling
    
    Args:
        wet_bulb_temp: Wet bulb temperature (°C)
        dry_bulb_temp: Dry bulb (air) temperature (°C)
        wind_speed_10m: 10m wind speed (m/s)
        
    Returns:
        WBGT outdoor estimated values (°C)
        
    Heat stress thresholds:
        > 32°C: Stop outdoor activity
        28-32°C: Extreme caution
        25-28°C: High caution  
        < 25°C: Generally safe
    """
    # Input validation
    if wet_bulb_temp is None or dry_bulb_temp is None or wind_speed_10m is None:
        raise ValueError("Temperature and wind inputs cannot be None")
    
    # Use moderate solar load (assumes daytime conditions for conservative estimate)
    # This provides a reasonable outdoor WBGT without needing specific forecast hour
    solar_factor = 2.0  # Moderate solar heating (2°C boost)
    
    # Estimate black globe temperature
    # Formula: BG = air_temp + solar_heating - wind_cooling
    black_globe = (dry_bulb_temp + 
                  solar_factor - 
                  (wind_speed_10m * 0.4))  # Wind cooling factor
    
    # Black globe cannot be cooler than air temperature
    black_globe = np.maximum(black_globe, dry_bulb_temp)
    
    # WBGT calculation with estimated black globe
    wbgt = (0.7 * wet_bulb_temp + 
            0.2 * black_globe + 
            0.1 * dry_bulb_temp)
    
    # Mask invalid data
    wbgt = np.where(
        (np.isnan(wet_bulb_temp)) | (np.isnan(dry_bulb_temp)) | (np.isnan(wind_speed_10m)),
        np.nan, wbgt
    )
    
    return wbgt
