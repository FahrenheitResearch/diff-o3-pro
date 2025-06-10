from .common import *

def seqouiagrove_thermal_range_index(temp_max: np.ndarray, temp_min: np.ndarray,
                                   dewpoint_2m: np.ndarray, wind_speed_10m: np.ndarray,
                                   cloud_cover: np.ndarray = None) -> np.ndarray:
    """
    Compute Seqouiagrove Thermal Range Index (STRI)
    
    Highlights areas with significant diurnal temperature spreads that create
    those magical Seqouiagrove moments - big daily temperature swings that 
    signal dynamic atmospheric conditions and potential for dramatic weather.
    
    Formula incorporates:
    - Base diurnal temperature range (°C)
    - Dryness enhancement (low dewpoint = bigger swings)
    - Wind modulation (light winds = bigger range, strong winds = mixing)
    - Cloud penalty (clear skies = bigger range)
    
    Args:
        temp_max: Daily maximum temperature (°C)
        temp_min: Daily minimum temperature (°C) 
        dewpoint_2m: 2m dewpoint temperature (°C)
        wind_speed_10m: 10m wind speed (m/s)
        cloud_cover: Total cloud cover (%) - optional
        
    Returns:
        STRI values (dimensionless)
    """
    # 1. Base diurnal temperature range
    temp_range = temp_max - temp_min
    
    # 2. Dryness enhancement factor
    # Low dewpoint = dry air = bigger potential temperature swings
    # Use dewpoint depression (T - Td) as dryness proxy
    # Convert temp_max to match dewpoint units if needed
    if np.mean(temp_max) > 200:  # Likely in Kelvin
        temp_max_c = temp_max - 273.15
    else:
        temp_max_c = temp_max
        
    dewpoint_depression = temp_max_c - dewpoint_2m
    dryness_factor = np.minimum(dewpoint_depression / 20.0, 2.0)  # Cap at 2x boost
    dryness_factor = np.maximum(dryness_factor, 0.5)  # Minimum 0.5x (never penalty)
    
    # 3. Wind modulation factor
    # Light winds (< 5 m/s) = less mixing = bigger ranges
    # Strong winds (> 15 m/s) = mixing = smaller ranges  
    wind_factor = np.where(wind_speed_10m < 5, 1.5,  # Boost for calm conditions
                          np.where(wind_speed_10m > 15, 0.7,  # Reduction for windy
                                  1.0 - (wind_speed_10m - 5) / 20.0))  # Linear between
    wind_factor = np.clip(wind_factor, 0.5, 1.5)
    
    # 4. Cloud cover penalty (if available)
    if cloud_cover is not None:
        # Clear skies = bigger ranges, overcast = smaller ranges
        cloud_factor = 1.0 - (cloud_cover / 100.0) * 0.4  # Up to 40% reduction
        cloud_factor = np.clip(cloud_factor, 0.6, 1.0)
    else:
        cloud_factor = 1.0  # No cloud data available
    
    # 5. Seqouiagrove Thermal Range Index
    # Base range enhanced by environmental factors
    stri = temp_range * dryness_factor * wind_factor * cloud_factor
    
    # Add bonus for exceptional ranges (>20°C gets extra credit)
    exceptional_bonus = np.where(temp_range > 20, (temp_range - 20) * 0.2, 0)
    stri += exceptional_bonus
    
    # Ensure non-negative
    stri = np.maximum(stri, 0)
    
    # Mask invalid data
    stri = np.where(
        (np.isnan(temp_max)) | (np.isnan(temp_min)) | (np.isnan(dewpoint_2m)) | 
        (np.isnan(wind_speed_10m)), 
        np.nan, stri
    )
    
    return stri
