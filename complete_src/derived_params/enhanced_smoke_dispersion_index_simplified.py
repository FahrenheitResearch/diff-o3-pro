from .common import *

def enhanced_smoke_dispersion_index_simplified(wind_shear: np.ndarray, temp_surface: np.ndarray,
                                              boundary_layer_height: np.ndarray,
                                              u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Compute Enhanced Smoke Dispersion Index (simplified version)
    """
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    # Use temperature as a simple stability proxy
    # High temperature = more unstable (negative stability for better mixing)
    stability = -(temp_surface - 288.15) / 20.0  # Normalize around 15°C
    
    return enhanced_smoke_dispersion_index(
        wind_shear, stability, boundary_layer_height, wind_speed
    )
