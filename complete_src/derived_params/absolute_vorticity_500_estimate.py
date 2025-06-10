from .common import *

def absolute_vorticity_500_estimate(wind_speed: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """
    Simple estimate of 500mb absolute vorticity from wind speed
    
    Args:
        wind_speed: 500mb wind speed (m/s)
        latitude: Latitude array (degrees)
        
    Returns:
        Absolute vorticity estimate (s⁻¹)
    """
    # Coriolis parameter
    omega = 7.2921e-5  # Earth's rotation rate (s⁻¹)
    f = 2 * omega * np.sin(np.radians(latitude))
    
    # Simple curvature vorticity estimate based on wind speed
    # Higher wind speeds in westerly flow suggest more curvature
    curvature_vort = wind_speed / 500000.0  # Rough scaling (s⁻¹)
    
    # Absolute vorticity = relative vorticity + Coriolis
    abs_vort = curvature_vort + f
    
    return np.abs(abs_vort)  # Take absolute value for composite
