from .common import *

def surface_based_cape_and_cin(temp_profile_k: np.ndarray, dewpoint_profile_k: np.ndarray,
                             pressure_profile_pa: np.ndarray, 
                             temp_surface_k: np.ndarray, dewpoint_surface_k: np.ndarray,
                             pressure_surface_pa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Surface-Based CAPE and CIN using proper thermodynamic formulas
    
    CAPE = g ∫[LCL to EL] (Tp - Te)/Te dz (only positive buoyancy)
    CIN = g ∫[surface to LFC] (Tp - Te)/Te dz (only negative buoyancy)
    
    Uses virtual temperature correction for accuracy
    
    Args:
        temp_profile_k: Temperature profile on pressure levels (K)
        dewpoint_profile_k: Dewpoint profile on pressure levels (K) 
        pressure_profile_pa: Pressure levels (Pa)
        temp_surface_k: Surface temperature (K)
        dewpoint_surface_k: Surface dewpoint (K)
        pressure_surface_pa: Surface pressure (Pa)
        
    Returns:
        Tuple of (SBCAPE in J/kg, SBCIN in J/kg)
    """
    # Constants
    g = 9.81  # m/s²
    Rd = 287.0  # J/kg/K
    
    # Calculate surface mixing ratio
    es_surface = _calculate_saturation_vapor_pressure(temp_surface_k)
    e_surface = _calculate_saturation_vapor_pressure(dewpoint_surface_k)
    mixing_ratio_surface = 0.622 * e_surface / (pressure_surface_pa / 100.0 - e_surface)  # kg/kg
    
    # Find LCL
    lcl_pressure_pa, lcl_temp_k = _find_lcl_bolton(
        temp_surface_k, dewpoint_surface_k, pressure_surface_pa)
    
    # Initialize CAPE and CIN
    cape = np.zeros_like(temp_surface_k)
    cin = np.zeros_like(temp_surface_k)
    
    # Process each vertical level
    found_lfc = np.zeros_like(temp_surface_k, dtype=bool)
    found_el = np.zeros_like(temp_surface_k, dtype=bool)
    
    for i in range(len(pressure_profile_pa)):
        p_level = pressure_profile_pa[i]
        
        # Skip levels above surface
        above_surface = p_level >= pressure_surface_pa
        
        # Calculate parcel temperature at this level
        below_lcl = p_level > lcl_pressure_pa
        
        # For levels below LCL, lift dry adiabatically
        parcel_temp_k = np.where(below_lcl,
            temp_surface_k * (p_level / pressure_surface_pa) ** (Rd / 1004.0),
            _moist_adiabatic_temperature(lcl_temp_k, lcl_pressure_pa, p_level))
        
        # Calculate virtual temperatures with moisture correction
        env_temp_k = temp_profile_k[i]
        
        # Environment mixing ratio
        es_env = _calculate_saturation_vapor_pressure(temp_profile_k[i])
        e_env = _calculate_saturation_vapor_pressure(dewpoint_profile_k[i])
        mixing_ratio_env = 0.622 * e_env / (p_level / 100.0 - e_env)
        
        # Parcel keeps surface mixing ratio below LCL, saturated above LCL
        parcel_mixing_ratio = np.where(below_lcl, 
            mixing_ratio_surface,
            0.622 * _calculate_saturation_vapor_pressure(parcel_temp_k) / 
            (p_level / 100.0 - _calculate_saturation_vapor_pressure(parcel_temp_k)))
        
        # Virtual temperatures
        parcel_tv = _calculate_virtual_temperature(parcel_temp_k, parcel_mixing_ratio)
        env_tv = _calculate_virtual_temperature(env_temp_k, mixing_ratio_env)
        
        # Buoyancy (only where above surface)
        buoyancy = (parcel_tv - env_tv) / env_tv
        buoyancy = np.where(above_surface, buoyancy, 0)
        
        # Calculate layer thickness (pressure to height conversion)
        if i < len(pressure_profile_pa) - 1:
            dp = pressure_profile_pa[i] - pressure_profile_pa[i + 1]
            dz = -Rd * env_tv * dp / (g * p_level)  # m
            
            # Positive buoyancy = CAPE (only above LCL)
            is_cape_layer = (buoyancy > 0) & (p_level <= lcl_pressure_pa) & ~found_el
            cape_contrib = g * buoyancy * dz
            cape = np.where(is_cape_layer, cape + cape_contrib, cape)
            
            # Track EL (first level where buoyancy becomes negative after being positive)
            was_positive = cape > 0
            becomes_negative = (buoyancy <= 0) & was_positive & ~found_el
            found_el = found_el | becomes_negative
            
            # Negative buoyancy = CIN (only below LFC)
            # LFC is first level where buoyancy becomes positive above LCL
            becomes_positive = (buoyancy > 0) & (p_level <= lcl_pressure_pa) & ~found_lfc
            found_lfc = found_lfc | becomes_positive
            
            is_cin_layer = (buoyancy < 0) & (p_level > lcl_pressure_pa) & ~found_lfc
            cin_contrib = g * buoyancy * dz
            cin = np.where(is_cin_layer, cin + cin_contrib, cin)
    
    # Ensure CAPE >= 0 and CIN <= 0
    cape = np.maximum(cape, 0.0)
    cin = np.minimum(cin, 0.0)
    
    # Set reasonable upper bounds
    cape = np.minimum(cape, 8000.0)  # Cap at 8000 J/kg
    cin = np.maximum(cin, -1000.0)  # Cap at -1000 J/kg
    
    return cape, cin
