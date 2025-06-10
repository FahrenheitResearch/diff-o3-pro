from .common import *

def samuel_outflow_propensity_simplified(temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                                       rh_500mb: np.ndarray, sbcape: np.ndarray = None) -> np.ndarray:
    """
    Simplified S-OPI when DCAPE is not directly available
    
    Estimates DCAPE from surface conditions and SBCAPE when needed
    
    Args:
        temp_2m: 2m temperature (°C or K)
        dewpoint_2m: 2m dewpoint temperature (°C)
        rh_500mb: 500mb relative humidity (%)
        sbcape: Surface-based CAPE (J/kg) - optional for DCAPE estimation
        
    Returns:
        S-OPI values (0-1 scale)
    """
    # Estimate DCAPE if not available
    # Simple approximation: DCAPE ≈ 0.3 * SBCAPE for dry boundary layers
    # This is a rough proxy based on typical Norman sounding characteristics
    if sbcape is not None:
        # Convert temperature if in Kelvin
        if np.mean(temp_2m) > 200:
            temp_2m_c = temp_2m - 273.15
        else:
            temp_2m_c = temp_2m
            
        # Estimate DCAPE based on dryness and instability
        dewpoint_depression = temp_2m_c - dewpoint_2m
        dryness_factor = np.minimum(dewpoint_depression / 20.0, 1.0)
        estimated_dcape = sbcape * 0.3 * dryness_factor
    else:
        # Very rough fallback estimate
        if np.mean(temp_2m) > 200:
            temp_2m_c = temp_2m - 273.15
        else:
            temp_2m_c = temp_2m
            
        dewpoint_depression = temp_2m_c - dewpoint_2m
        # Estimate based on temperature and dryness
        estimated_dcape = dewpoint_depression * 50.0  # Very rough proxy
    
    return samuel_outflow_propensity_index(
        estimated_dcape, temp_2m, dewpoint_2m, rh_500mb
    )
