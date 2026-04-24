import numpy as np
import cv2
from afqa_toolbox.features import FeatGabor # type: ignore
from .helper import _finalize
# from your_library import FeatGabor

def compute_gabor(image, mask, **kwargs):
    blk_size  = kwargs.get("gabor_blk_size", 16)
    sigma     = kwargs.get("gabor_sigma", 6)
    freq      = kwargs.get("gabor_freq", 0.1)
    angle_num = kwargs.get("gabor_angle_num", 8)

    feat = FeatGabor(blk_size=blk_size, sigma=sigma, freq=freq, angle_num=angle_num)
    std_map = feat.gabor_stds(image, smooth=True, shen=False)

   

    return _finalize(std_map, mask)

        
