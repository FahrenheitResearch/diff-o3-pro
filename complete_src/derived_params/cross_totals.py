from .common import *

def cross_totals(temp_850: np.ndarray, dewpoint_850: np.ndarray, 
                temp_500: np.ndarray) -> np.ndarray:
    """
    Compute Cross Totals index
    
    Args:
        temp_850: 850mb temperature (°C)
        dewpoint_850: 850mb dewpoint (°C)
        temp_500: 500mb temperature (°C)
        
    Returns:
        Cross Totals (°C)
    """
    return dewpoint_850 - temp_500
