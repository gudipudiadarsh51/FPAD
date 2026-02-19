import numpy as np

def compute_mean(foreground_cropped):
    img_float = foreground_cropped.astype(np.float64)                 # Mean

    # Mean intensity
    mean_intensity = float(np.mean(img_float))

    return mean_intensity