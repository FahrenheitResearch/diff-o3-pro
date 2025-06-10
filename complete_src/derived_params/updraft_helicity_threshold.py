from .common import *

def updraft_helicity_threshold(uh_data: np.ndarray, 
                             threshold: float = 75.0) -> np.ndarray:
    """
    Create binary mask for updraft helicity exceeding threshold
    
    Args:
        uh_data: Updraft helicity values (m²/s²)
        threshold: Threshold value (default 75 m²/s² for tornado potential)
        
    Returns:
        Binary mask (1 where UH >= threshold, 0 elsewhere)
    """
    return np.where(uh_data >= threshold, 1, 0)
