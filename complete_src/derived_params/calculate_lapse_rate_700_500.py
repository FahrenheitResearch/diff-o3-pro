"""
700-500mb Lapse Rate Calculation

Calculates the environmental lapse rate between 700mb and 500mb pressure levels.
Used in the Significant Hail Parameter (SHIP) calculation.

Formula: Lapse Rate = (T700 - T500) / (Z500 - Z700) * 1000
where temperatures are in Celsius and heights yield lapse rate in °C/km
"""

import numpy as np

def calculate_lapse_rate_700_500(temp_700, temp_500):
    """
    Calculate 700-500mb lapse rate for SHIP calculation.
    
    Parameters:
    -----------
    temp_700 : numpy.ndarray
        700mb temperature in Celsius
    temp_500 : numpy.ndarray  
        500mb temperature in Celsius
        
    Returns:
    --------
    numpy.ndarray
        Lapse rate in °C/km
        
    Notes:
    ------
    Uses standard atmosphere thickness between 700-500mb:
    - 700mb: ~3000m
    - 500mb: ~5500m
    - Thickness: ~2500m = 2.5km
    
    Typical values: 6-8°C/km for unstable atmosphere
    """
    
    # Standard atmospheric thickness between 700-500mb levels
    # This is an approximation - ideally we'd use actual heights
    thickness_km = 2.5  # ~2500m between 700-500mb
    
    # Calculate lapse rate: (warm - cold) / thickness
    lapse_rate = (temp_700 - temp_500) / thickness_km
    
    # Ensure reasonable values (3-12°C/km typical range)
    lapse_rate = np.clip(lapse_rate, 0, 15)
    
    return lapse_rate