# Standard deviation of intensity
import numpy as np

def compute_std(img_float):
    std_intensity = float(np.std(img_float, ddof=1))                # standard devation
    return std_intensity