from .common import *

def lapse_rate_03km(temp_surface: np.ndarray, temp_700: np.ndarray,
                   height_surface: np.ndarray, height_700: np.ndarray,
                   height_profile: Optional[np.ndarray] = None,
                   temp_profile: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute 0-3 km environmental lapse rate using height-based interpolation
    
    Expert fixes: 
    1. Prefer full profile interpolation with MetPy when available (robust)
    2. Fall back to 2-level linear interpolation with sanity checks 
    3. Use height-based calculation (surface to 3km AGL) instead of 
       pressure-based (surface to 700mb) to match SPC/Pivotal methodology
    
    Following the four critical checks from HRRR lapse rate debugging guide:
    1. Use actual grid-point thickness, not constant
    2. Divide by kilometers, not meters 
    3. Subtract in correct order (warm - cold)
    4. Use appropriate contour levels
    
    Args:
        temp_surface: Surface temperature (K)
        temp_700: Temperature at 700mb (K) - used for interpolation
        height_surface: Surface geopotential height (m)
        height_700: 700mb geopotential height (m)
        height_profile: Full height profile (m) for MetPy interpolation [optional]
        temp_profile: Full temperature profile (K) for MetPy interpolation [optional]
        
    Returns:
        0-3 km lapse rate (°C/km)
    """
    # EXPERT IMPLEMENTATION: Prefer full profile interpolation with MetPy
    if (METPY_AVAILABLE and height_profile is not None and temp_profile is not None):
        # PREFERRED: Use MetPy interpolation with full atmospheric profile
        try:
            from metpy.calc import interpolate_1d
            # Convert to MetPy units if needed
            h_units = height_profile * units.m if not hasattr(height_profile, 'units') else height_profile
            t_units = temp_profile * units.K if not hasattr(temp_profile, 'units') else temp_profile
            h_sfc_units = height_surface * units.m if not hasattr(height_surface, 'units') else height_surface
            
            # Target height: surface + 3000m
            target_height_3km = h_sfc_units + 3000 * units.m
            
            # Interpolate temperature to exactly 3km AGL using full profile
            temp_3km = interpolate_1d(target_height_3km, h_units, t_units, axis=0)
            
            # Convert surface temperature to Celsius with units
            if temp_surface.max() > 200:  # Likely in Kelvin
                temp_surface_c = (temp_surface - 273.15)
            else:
                temp_surface_c = temp_surface
                
            # Convert 3km temperature to Celsius
            temp_3km_c = temp_3km.to('degC').magnitude if hasattr(temp_3km, 'units') else temp_3km - 273.15
            
            # Calculate lapse rate: (T_surface - T_3km) / 3km
            lapse_rate = (temp_surface_c - temp_3km_c) / 3.0
            
            print(f"✅ Using MetPy profile interpolation for 0-3km lapse rate")
            
        except Exception as e:
            print(f"⚠️ MetPy lapse rate interpolation failed: {e}")
            # Fall back to 2-level approach
            lapse_rate = _compute_2level_lapse_rate(temp_surface, temp_700, height_surface, height_700)
    else:
        # FALLBACK: 2-level linear interpolation with expert sanity checks
        lapse_rate = _compute_2level_lapse_rate(temp_surface, temp_700, height_surface, height_700)
    
    # Debug output for validation
    valid_count = np.isfinite(lapse_rate).sum()
    if valid_count > 0:
        print(f"🔍 Lapse rate debug: {np.nanmin(lapse_rate):.2f}-{np.nanmax(lapse_rate):.2f}°C/km ({valid_count} valid points)")
    else:
        print(f"🔍 Lapse rate debug: No valid points!")
    
    # SPC-based sanity check for 0-3km lapse rates:
    # Expected distribution (expert guidance):
    # 5th percentile: ~4.5°C/km, 50th: ~6.5°C/km, 95th: ~9.0°C/km
    # < 5.5-6.0°C/km: Stable (moist adiabatic)
    # 5.5-9.8°C/km: Conditionally unstable  
    # > 9.8°C/km: Absolutely unstable (dry adiabatic limit)
    # Clip to meteorologically realistic range per SPC guidance
    lapse_rate = np.where(np.isfinite(lapse_rate), 
                         np.clip(lapse_rate, 2.0, 10.0),  # Cap at dry adiabatic
                         np.nan)
    
    print(f"   Final clipped: {np.nanmin(lapse_rate):.2f}-{np.nanmax(lapse_rate):.2f}°C/km")
    
    return lapse_rate

def _compute_2level_lapse_rate(temp_surface: np.ndarray, temp_700: np.ndarray,
                              height_surface: np.ndarray, height_700: np.ndarray) -> np.ndarray:
    """2-level fallback with expert-recommended sanity checks"""
    
    # Convert temperatures to Celsius
    if temp_surface.max() > 200:  # Likely in Kelvin
        temp_surface_c = temp_surface - 273.15
        temp_700_c = temp_700 - 273.15
    else:  # Already in Celsius
        temp_surface_c = temp_surface
        temp_700_c = temp_700
    
    # EXPERT SANITY CHECK: Ensure 700mb is sufficiently above surface for interpolation
    height_diff_total = height_700 - height_surface
    
    # Expert recommendation: if 700mb < surface + 1500m, mark as invalid
    # This prevents bad interpolation in deep low-pressure areas
    valid_thickness = height_diff_total > 1500  # Must have at least 1.5km between levels
    
    # HEIGHT-BASED APPROACH: Calculate temperature at exactly 3km AGL
    target_height_3km = height_surface + 3000.0
    height_diff_to_3km = target_height_3km - height_surface
    
    # Interpolation factor (0 = surface, 1 = 700mb level)
    # Avoid division by zero where thickness is invalid
    interp_factor = np.where(valid_thickness, 
                            height_diff_to_3km / height_diff_total, 
                            np.nan)
    
    # Interpolated temperature at 3km AGL
    temp_3km_c = temp_surface_c + interp_factor * (temp_700_c - temp_surface_c)
    
    # Calculate lapse rate: (T_surface - T_3km) / 3km
    lapse_rate = np.where(valid_thickness, 
                         (temp_surface_c - temp_3km_c) / 3.0,
                         np.nan)
    
    print(f"📊 Using 2-level interpolation fallback")
    print(f"   Valid thickness points: {valid_thickness.sum()} of {valid_thickness.size}")
    
    return lapse_rate
