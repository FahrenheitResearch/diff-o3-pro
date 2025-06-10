from .common import *
from .fire_weather_index import fire_weather_index

def fire_weather_index_from_components(temp: np.ndarray, humidity: np.ndarray,
                                     u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Compute Fire Weather Index from wind components
    """
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    return fire_weather_index(temp, humidity, wind_speed)
