import numpy as np
import cv2
from scipy.stats import skew

def _finalize(feature_map, mask=None, threshold=0.5):
    """
    Aggregate block-level feature map safely
    """

    # 🔥 Handle mask mismatch
    if mask is not None:
        if feature_map.shape != mask.shape:
            # resize mask to match feature_map
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (feature_map.shape[1], feature_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask

        values = feature_map[mask_resized > 0]
    else:
        values = feature_map.flatten()

    # remove NaNs
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return [0.0] * 6

    
    return [
        np.mean(values),
        np.std(values)
    ]