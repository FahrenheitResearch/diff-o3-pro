from .common import *

def calculate_surface_based_cape(temp_2m: np.ndarray, dewpoint_2m: np.ndarray, 
                               pressure_surface: np.ndarray) -> np.ndarray:
    """
    Calculate Surface-Based CAPE for derived parameter system
    
    Note: This is a simplified version that estimates CAPE without full profile data.
    For full accuracy, use surface_based_cape_and_cin with complete sounding data.
    
    Args:
        temp_2m: 2m temperature (K)
        dewpoint_2m: 2m dewpoint (K)
        pressure_surface: Surface pressure (Pa)
        
    Returns:
        SBCAPE estimate (J/kg)
    """
    # Simplified CAPE estimation using Bolton (1980) approximation
    # This provides a reasonable estimate when full profile data isn't available
    
    # Convert to Celsius
    temp_c = temp_2m - 273.15
    dewpoint_c = dewpoint_2m - 273.15
    
    # Estimate CAPE using Bolton's approximation
    # CAPE ≈ (Cp * T) * ln(θe_parcel / θe_environment)
    
    # Calculate surface equivalent potential temperature
    es_surface = _calculate_saturation_vapor_pressure(temp_2m)
    e_surface = _calculate_saturation_vapor_pressure(dewpoint_2m)
    mixing_ratio = 0.622 * e_surface / (pressure_surface / 100.0 - e_surface)
    
    # Potential temperature
    theta = temp_2m * (100000.0 / pressure_surface) ** 0.286
    
    # Equivalent potential temperature (simplified)
    theta_e_surface = theta * np.exp(2.5e6 * mixing_ratio / (1004.0 * temp_2m))
    
    # Estimate environmental θe (assuming typical atmospheric profile)
    # Use a standard atmospheric lapse to estimate 500mb conditions
    temp_500_est = temp_2m - 0.0065 * 5500  # Rough 500mb height
    theta_e_500_est = temp_500_est * (100000.0 / 50000.0) ** 0.286
    
    # CAPE approximation
    cape_est = 1004.0 * temp_2m * np.log(theta_e_surface / theta_e_500_est)
    cape_est = np.maximum(cape_est, 0.0)  # No negative CAPE
    cape_est = np.minimum(cape_est, 8000.0)  # Cap at reasonable max
    
    return cape_est
