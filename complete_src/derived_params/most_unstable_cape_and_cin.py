from .common import *

def most_unstable_cape_and_cin(temp_profile_k: np.ndarray, dewpoint_profile_k: np.ndarray,
                             pressure_profile_pa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Most-Unstable CAPE and CIN using parcel with highest θe in lowest 300 mb
    
    Args:
        temp_profile_k: Temperature profile on pressure levels (K)
        dewpoint_profile_k: Dewpoint profile on pressure levels (K)
        pressure_profile_pa: Pressure levels (Pa)
        
    Returns:
        Tuple of (MUCAPE in J/kg, MUCIN in J/kg)
    """
    # Find surface level (highest pressure)
    surface_idx = np.argmax(pressure_profile_pa)
    pressure_surface_pa = pressure_profile_pa[surface_idx]
    
    # Search lowest 300 mb
    search_top_pa = pressure_surface_pa - 30000  # 300 mb = 30000 Pa
    in_search_layer = pressure_profile_pa >= search_top_pa
    search_indices = np.where(in_search_layer)[0]
    
    if len(search_indices) < 1:
        # Use surface if no levels found
        return surface_based_cape_and_cin(
            temp_profile_k, dewpoint_profile_k, pressure_profile_pa,
            temp_profile_k[surface_idx], dewpoint_profile_k[surface_idx],
            pressure_profile_pa[surface_idx]
        )
    
    # Calculate equivalent potential temperature at each level
    max_theta_e = -999.0
    mu_level_idx = surface_idx
    
    for idx in search_indices:
        temp_k = temp_profile_k[idx]
        dewpoint_k = dewpoint_profile_k[idx]
        pressure_pa = pressure_profile_pa[idx]
        
        # Simple equivalent potential temperature approximation
        # θe ≈ θ * exp(Lv * r / (Cp * T))
        theta = temp_k * (100000.0 / pressure_pa) ** 0.286  # Potential temperature
        
        # Mixing ratio
        es = _calculate_saturation_vapor_pressure(dewpoint_k)
        r = 0.622 * es / (pressure_pa / 100.0 - es)
        
        # Equivalent potential temperature (simplified)
        theta_e = theta * np.exp(2.5e6 * r / (1004.0 * temp_k))
        
        if theta_e > max_theta_e:
            max_theta_e = theta_e
            mu_level_idx = idx
    
    # Use most unstable parcel
    return surface_based_cape_and_cin(
        temp_profile_k, dewpoint_profile_k, pressure_profile_pa,
        temp_profile_k[mu_level_idx], dewpoint_profile_k[mu_level_idx],
        pressure_profile_pa[mu_level_idx]
    )
