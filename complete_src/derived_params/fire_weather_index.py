from .common import *

def fire_weather_index(temp: np.ndarray, humidity: np.ndarray,
                      wind_speed: np.ndarray, precipitation: np.ndarray = None) -> np.ndarray:
    """
    Compute simplified Fire Weather Index
    
    Args:
        temp: Temperature (°C)
        humidity: Relative humidity (%)
        wind_speed: Wind speed (m/s)
        precipitation: Precipitation (mm) - optional
        
    Returns:
        Fire Weather Index (dimensionless)
    """
    # Normalize components
    temp_factor = np.maximum((temp - 10) / 30.0, 0)  # Above 10°C
    humidity_factor = np.maximum((80 - humidity) / 60.0, 0)  # Below 80%
    wind_factor = np.minimum(wind_speed / 15.0, 2.0)  # Up to 15 m/s
    
    # Base index
    fwi = temp_factor * humidity_factor * wind_factor
    
    # Precipitation reduction
    if precipitation is not None:
        precip_factor = np.where(precipitation > 1.0, 0.5, 1.0)
        fwi *= precip_factor
    
    return np.clip(fwi, 0, 10)
