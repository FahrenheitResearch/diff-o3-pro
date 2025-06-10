from .common import *

def calculate_surface_based_cin(temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                              pressure_surface: np.ndarray) -> np.ndarray:
    """
    Calculate Surface-Based CIN for derived parameter system
    
    Args:
        temp_2m: 2m temperature (K)
        dewpoint_2m: 2m dewpoint (K) 
        pressure_surface: Surface pressure (Pa)
        
    Returns:
        SBCIN estimate (J/kg, negative values)
    """
    # Simplified CIN estimation based on surface conditions
    # Estimate cap strength from temperature-dewpoint spread
    
    temp_c = temp_2m - 273.15
    dewpoint_c = dewpoint_2m - 273.15
    
    # Dewpoint depression as proxy for atmospheric dryness/stability
    dewpoint_depression = temp_c - dewpoint_c
    
    # Estimate CIN based on typical cap strength relationships
    # Strong dewpoint depression often indicates capping inversion
    cin_est = -10.0 * dewpoint_depression  # J/kg per °C depression
    
    # Apply seasonal/thermal adjustments
    # Hot surface temperatures with dry air aloft = stronger cap
    temp_factor = np.where(temp_c > 30, 1.5, 1.0)
    cin_est = cin_est * temp_factor
    
    # Ensure CIN is negative and within reasonable bounds
    cin_est = np.minimum(cin_est, 0.0)  # No positive CIN
    cin_est = np.maximum(cin_est, -500.0)  # Cap at reasonable magnitude
    
    # Low dewpoint depression = little or no cap
    cin_est = np.where(dewpoint_depression < 5, 0.0, cin_est)
    
    return cin_est
