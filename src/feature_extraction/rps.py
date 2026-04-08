from afqa_toolbox.features import FeatRPS                             # type: ignore # Radial Power Spectrum
import pandas as pd
import numpy as np

def compute_rps(foreground_cropped: np.ndarray) -> float:
    rps_feat = FeatRPS()
    rps_value = rps_feat.rps(foreground_cropped)

    return float(rps_value)
