from .common import *

def samuel_outflow_propensity_index(dcape: np.ndarray, temp_2m: np.ndarray,
                                  dewpoint_2m: np.ndarray, rh_500mb: np.ndarray) -> np.ndarray:
    """
    Compute Samuel Outflow Propensity Index (S-OPI)
    
    A science-first composite that quantifies how likely storms are to go 
    "cold-pool dominant" in the Norman, OK corridor. Uses three key physical
    drivers of strong outflow.
    
    Formula: S-OPI = D × E × M
    Where:
    - D = min(DCAPE/1500, 1) - Downdraft CAPE normalized
    - E = min((T2-Td2)/25, 1) - Evaporative cooling potential  
    - M = RH500/100 - Mid-level moisture for precipitation loading
    
    Args:
        dcape: Downdraft CAPE (J/kg)
        temp_2m: 2m temperature (°C or K)
        dewpoint_2m: 2m dewpoint temperature (°C)
        rh_500mb: 500mb relative humidity (%)
        
    Returns:
        S-OPI values (0-1 scale, dimensionless)
    """
    # Convert temperature if in Kelvin
    if np.mean(temp_2m) > 200:  # Likely in Kelvin
        temp_2m_c = temp_2m - 273.15
    else:
        temp_2m_c = temp_2m
    
    # 1. Downdraft CAPE term: DCAPE/1500 J/kg (clip ≤ 1)
    # DCAPE ≈ 1500 J/kg marks 90th percentile "big microburst" days in Norman
    dcape_term = np.minimum(dcape / 1500.0, 1.0)
    dcape_term = np.maximum(dcape_term, 0.0)
    
    # 2. Evaporative cooling term: (T2-Td2)/25°C (clip ≤ 1)
    # Dewpoint depression of 25°C is extreme dry layer; 10-15°C common
    # Dry low levels maximize raindrop evaporation and cooling
    dewpoint_depression = temp_2m_c - dewpoint_2m
    evap_term = np.minimum(dewpoint_depression / 25.0, 1.0)
    evap_term = np.maximum(evap_term, 0.0)
    
    # 3. Mid-level moisture term: RH500/100% (0-1 scale)
    # Moist mid-troposphere supplies condensate that can fall, evaporate,
    # and accelerate the downdraft
    moisture_term = rh_500mb / 100.0
    moisture_term = np.clip(moisture_term, 0.0, 1.0)
    
    # Samuel Outflow Propensity Index (multiplicative!)
    # One weak link (e.g., very moist surface) crushes the score
    s_opi = dcape_term * evap_term * moisture_term
    
    # Ensure 0-1 bounds
    s_opi = np.clip(s_opi, 0.0, 1.0)
    
    # Mask invalid data
    s_opi = np.where(
        (np.isnan(dcape)) | (np.isnan(temp_2m)) | (np.isnan(dewpoint_2m)) | 
        (np.isnan(rh_500mb)),
        np.nan, s_opi
    )
    
    return s_opi
