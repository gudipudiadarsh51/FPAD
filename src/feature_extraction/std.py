# Standard deviation of intensity
import numpy as np

def compute_std(image, mask, **kwargs):
    if mask is not None:
        if np.sum(mask) == 0:
            return 0.0, 0.0
        values = image[mask > 0]
    else:
        values = image.flatten()

    return float(np.std(values)), 0.0