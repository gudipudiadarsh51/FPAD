import numpy as np
import cv2
from afqa_toolbox.features import FeatGabor # type: ignore


def compute_gabor(image):
        
    gabor_feat = FeatGabor(
        blk_size=16,
        sigma=6,
        freq=0.1,
        angle_num=8
    )

    std_map = gabor_feat.gabor_stds(
        image,
        smooth=False,
        shen=False
    )

    if std_map is None or std_map.size == 0:
        return 0.0, 0.0
    

    return float(np.nanmean(std_map)), float(np.nanstd(std_map))