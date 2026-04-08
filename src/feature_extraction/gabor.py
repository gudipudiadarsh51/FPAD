import numpy as np
import cv2
from afqa_toolbox.features import FeatGabor # type: ignore


def compute_gabor(image: np.ndarray, config: dict) -> tuple:
        
    gabor_feat = FeatGabor(
        blk_size=config['blk_size'],
        sigma=config['gabor_sigma'],
        freq=config['gabor_freq'],
        angle_num=config['gabor_angle_num']
    )

    std_map = gabor_feat.gabor_stds(
        image,
        smooth=True,
        shen=True
    )

    if std_map is None or std_map.size == 0:
        return 0.0, 0.0
    

    return float(np.nanmean(std_map)), float(np.nanstd(std_map))