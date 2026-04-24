from afqa_toolbox.features import FeatRPS                             # type: ignore # Radial Power Spectrum
import numpy as np

def compute_rps(image, mask, **kwargs):
    spectrum = np.abs(np.fft.fft2(image))

    if mask is not None:
        if np.sum(mask) == 0:
            return 0.0, 0.0
        

    values = spectrum.flatten()

    if values.size == 0:
        return 0.0, 0.0

    return float(np.mean(values)), float(np.std(values))
