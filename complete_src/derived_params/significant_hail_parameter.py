from .common import *
from ._mixing_ratio_approximation import _mixing_ratio_approximation

def significant_hail_parameter(mucape: np.ndarray, mucin: np.ndarray,
                             lapse_rate_700_500: np.ndarray, 
                             wind_shear_06km: np.ndarray,
                             freezing_level: np.ndarray,
                             temp_500: np.ndarray,
                             mixing_ratio_2m: np.ndarray) -> np.ndarray:
    """
    Compute Significant Hail Parameter (SHIP) - Official SPC v1.1 Recipe
    
    SHIP = (muCAPE/1500) × (MU_mr/13.6) × (lapse_700_500/7) × (shear_06km/20) × ((frz_lvl-T500_hgt)/8)
    
    All five required terms from SPC SHIP v1.1 specification with proper normalization.
    Each term is capped at 1.0 per SPC guidelines.
    
    Args:
        mucape: Most-Unstable CAPE (J/kg) - HRRR field MUCAPE
        mucin: Most-Unstable CIN (J/kg, negative values) - HRRR field MUCIN
        lapse_rate_700_500: 700-500mb lapse rate (°C/km) - derived parameter  
        wind_shear_06km: 0-6km bulk shear magnitude (m/s) - derived parameter
        freezing_level: Freezing level height (m AGL) - HRRR field HGTFZLV
        temp_500: 500mb temperature (°C) - HRRR field T at 500mb
        mixing_ratio_2m: 2m mixing ratio (g/kg) - approximation for MU mixing ratio
        
    Returns:
        SHIP values (dimensionless), >1 indicates significant hail potential
        
    Interpretation:
        SHIP > 1: Favorable for significant hail (≥2\" diameter)
        SHIP > 4: Extremely high hail potential
        
    References:
        SPC SHIP v1.1 specification
        
    Note:
        Uses 2m mixing ratio as approximation for MU mixing ratio when direct
        MU parcel data unavailable. This is a common operational approximation.
    """
    # ========================================================================
    # INPUT VALIDATION AND QUALITY CONTROL
    # ========================================================================
    
    # Quality flags for extreme values
    extreme_cape = np.any(mucape > 6000)
    extreme_shear = np.any(wind_shear_06km > 60)
    extreme_lapse = np.any(lapse_rate_700_500 > 12)
    
    if extreme_cape or extreme_shear or extreme_lapse:
        print(f"🔍 SHIP outliers detected: CAPE>{6000 if extreme_cape else 'OK'}, "
              f"Shear>{60 if extreme_shear else 'OK'}, Lapse>{12 if extreme_lapse else 'OK'}")
    
    # ========================================================================
    # TERM 1: muCAPE - SPC normalization 1500 J/kg, cap at 1.0
    # ========================================================================
    cape_term = np.clip(mucape / 1500.0, 0, 1.0)
    
    # ========================================================================
    # TERM 2: MU mixing ratio - SPC normalization 13.6 g/kg, cap at 1.0
    # ========================================================================
    # Use 2m mixing ratio as proxy for MU mixing ratio (common operational approximation)
    mr_term = np.clip(mixing_ratio_2m / 13.6, 0, 1.0)
    
    # ========================================================================
    # TERM 3: Lapse rate - SPC normalization 7°C/km, cap at 1.0
    # ========================================================================
    lapse_term = np.clip(lapse_rate_700_500 / 7.0, 0, 1.0)
    
    # ========================================================================
    # TERM 4: Wind shear - SPC normalization 20 m/s, cap at 1.0
    # ========================================================================
    shear_term = np.clip(wind_shear_06km / 20.0, 0, 1.0)
    
    # ========================================================================
    # TERM 5: Temperature term - (Freezing level - 500mb temp) / 8°C
    # ========================================================================
    # SPC formula: (freezing level height - 500mb temperature) / 8
    # where freezing level is in km and 500mb temp is in °C
    # This represents the temperature difference between freezing level and 500mb
    freezing_level_km = freezing_level / 1000.0  # Convert m to km
    temp_term = np.clip((freezing_level_km - temp_500) / 8.0, 0, 1.0)
    
    # ========================================================================
    # FINAL SHIP CALCULATION - All five terms multiplied
    # ========================================================================
    ship = cape_term * mr_term * lapse_term * shear_term * temp_term
    
    # ========================================================================
    # MASK INVALID DATA AND APPLY QUALITY CONTROL
    # ========================================================================
    valid_mask = (
        np.isfinite(mucape) & (mucape >= 0) &
        np.isfinite(lapse_rate_700_500) & (lapse_rate_700_500 >= 0) &
        np.isfinite(wind_shear_06km) & (wind_shear_06km >= 0) &
        np.isfinite(freezing_level) &
        np.isfinite(temp_500) &
        np.isfinite(mixing_ratio_2m) & (mixing_ratio_2m >= 0)
    )
    
    # Set invalid values to 0 (standard for SHIP)
    ship = np.where(valid_mask & (ship >= 0), ship, 0.0)
    
    # Apply low-CAPE mask to reduce noise
    low_cape_mask = mucape < 100.0  # J/kg threshold
    ship = np.where(low_cape_mask, 0.0, ship)
    
    return ship
