import numpy as np
import cv2
from afqa_toolbox.features import FeatGabor # type: ignore


def compute_gabor(image,blk_size=16,sigma=6,freq=0.1,angle_num=8):
        
    gabor_feat = FeatGabor(
        blk_size=blk_size,
        sigma=sigma,
        freq=freq,
        angle_num=angle_num
    )

    std_map = gabor_feat.gabor_stds(
        image,
        smooth=True,
        shen=False
    )

    if std_map is None or std_map.size == 0:
        return 0.0, 0.0
    

    return float(np.nanmean(std_map)), float(np.nanstd(std_map))