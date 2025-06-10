from .common import *

def seqouiagrove_thermal_range_simplified(temp_surface: np.ndarray, dewpoint_2m: np.ndarray,
                                        u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Simplified Seqouiagrove Thermal Range using single temperature snapshot
    
    Estimates diurnal potential from current conditions:
    - Uses surface temp as proxy for daily range potential
    - Dewpoint depression indicates dryness
    - Wind speed affects mixing
    
    Args:
        temp_surface: Current surface temperature (K or °C)
        dewpoint_2m: 2m dewpoint temperature (°C)
        u_wind: U wind component (m/s)
        v_wind: V wind component (m/s)
        
    Returns:
        Estimated thermal range index (dimensionless)
    """
    # Convert temperature if in Kelvin
    if np.mean(temp_surface) > 200:
        temp_c = temp_surface - 273.15
    else:
        temp_c = temp_surface
    
    # Calculate wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    # Estimate potential diurnal range based on current conditions
    # Higher current temp = more potential for big swings
    temp_potential = np.maximum(temp_c - 10, 0) / 5.0  # Normalize above 10°C
    
    # Use dewpoint depression as dryness proxy
    dewpoint_depression = temp_c - dewpoint_2m
    dryness_boost = np.minimum(dewpoint_depression / 15.0, 2.0)
    dryness_boost = np.maximum(dryness_boost, 0.3)
    
    # Wind mixing factor
    wind_factor = np.where(wind_speed < 3, 1.3,
                          np.where(wind_speed > 12, 0.7, 
                                  1.0 - (wind_speed - 3) / 15.0))
    wind_factor = np.clip(wind_factor, 0.5, 1.3)
    
    # Estimated thermal range index
    thermal_index = temp_potential * dryness_boost * wind_factor
    
    # Add seasonal boost (higher absolute temps = more potential)
    seasonal_boost = np.maximum((np.abs(temp_c) - 15) / 30.0, 0)
    thermal_index += seasonal_boost
    
    # Ensure reasonable bounds
    thermal_index = np.clip(thermal_index, 0, 15)
    
    # Mask invalid data
    thermal_index = np.where(
        (np.isnan(temp_surface)) | (np.isnan(dewpoint_2m)) | 
        (np.isnan(u_wind)) | (np.isnan(v_wind)),
        np.nan, thermal_index
    )
    
    return thermal_index
