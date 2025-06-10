from .common import *

def mixed_layer_cape_and_cin(temp_profile_k: np.ndarray, dewpoint_profile_k: np.ndarray,
                           pressure_profile_pa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Mixed-Layer CAPE and CIN using 100 mb mixed layer parcel
    
    Args:
        temp_profile_k: Temperature profile on pressure levels (K)
        dewpoint_profile_k: Dewpoint profile on pressure levels (K)
        pressure_profile_pa: Pressure levels (Pa)
        
    Returns:
        Tuple of (MLCAPE in J/kg, MLCIN in J/kg)
    """
    # Find surface level (highest pressure)
    surface_idx = np.argmax(pressure_profile_pa)
    pressure_surface_pa = pressure_profile_pa[surface_idx]
    
    # Define mixed layer (lowest 100 mb)
    mixed_layer_top_pa = pressure_surface_pa - 10000  # 100 mb = 10000 Pa
    
    # Find levels within mixed layer
    in_mixed_layer = pressure_profile_pa >= mixed_layer_top_pa
    mixed_indices = np.where(in_mixed_layer)[0]
    
    if len(mixed_indices) < 2:
        # Not enough levels for mixed layer, use surface
        return surface_based_cape_and_cin(
            temp_profile_k, dewpoint_profile_k, pressure_profile_pa,
            temp_profile_k[surface_idx], dewpoint_profile_k[surface_idx],
            pressure_profile_pa[surface_idx]
        )
    
    # Average temperature and dewpoint over mixed layer (pressure-weighted)
    ml_pressures = pressure_profile_pa[mixed_indices]
    ml_temps = temp_profile_k[mixed_indices]
    ml_dewpoints = dewpoint_profile_k[mixed_indices]
    
    # Pressure-weighted averages
    pressure_weights = ml_pressures / np.sum(ml_pressures)
    ml_temp_avg = np.sum(ml_temps * pressure_weights)
    ml_dewpoint_avg = np.sum(ml_dewpoints * pressure_weights)
    
    # Use surface pressure for parcel
    return surface_based_cape_and_cin(
        temp_profile_k, dewpoint_profile_k, pressure_profile_pa,
        ml_temp_avg, ml_dewpoint_avg, pressure_surface_pa
    )
