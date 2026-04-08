# Standard deviation of intensity
import numpy as np

def compute_std(img: np.ndarray) -> float:
    std_intensity = float(np.std(img, ddof=1))                # standard devation
    return std_intensity