from .common import *

def haines_index(temp_850: np.ndarray, temp_700: np.ndarray,
                dewpoint_850: np.ndarray, dewpoint_700: np.ndarray) -> np.ndarray:
    """
    Compute Haines Index for fire weather
    
    Args:
        temp_850: 850mb temperature (°C)
        temp_700: 700mb temperature (°C)
        dewpoint_850: 850mb dewpoint (°C)
        dewpoint_700: 700mb dewpoint (°C)
        
    Returns:
        Haines Index (2-6 scale)
    """
    # Stability term (A)
    stability = temp_850 - temp_700
    A = np.where(stability < 4, 1,
                np.where(stability < 8, 2, 3))
    
    # Moisture term (B)  
    moisture = temp_850 - dewpoint_850
    B = np.where(moisture < 6, 1,
                np.where(moisture < 10, 2, 3))
    
    haines = A + B
    
    return np.clip(haines, 2, 6)
