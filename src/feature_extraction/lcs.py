# src/feature_extraction/lcs.py

import numpy as np
import cv2

from afqa_toolbox.features import ( # type: ignore
    slanted_block_properties,
    covcoef,
    orient,FeatLCS
)
from .helper import _finalize

# Make sure FeatLCS is imported properly
# from your_library import FeatLCS


def compute_lcs(img, mask, **kwargs):
    blk_size = kwargs.get('shared_blk_size', 32)
    v1sz_x = min(kwargs.get('v1sz_x', blk_size), blk_size)
    v1sz_y = min(kwargs.get('v1sz_y', blk_size // 2), blk_size)
    foreground_ratio = kwargs.get('foreground_ratio', 0.8)

    

    feat = FeatLCS(blk_size=blk_size, v1sz_x=v1sz_x, v1sz_y=v1sz_y,foreground_ratio=foreground_ratio)  # Initialize the feature extractor
    ocl_map = feat.lcs(img, mask)  # Compute the LCS standard deviation map



    return _finalize(ocl_map, mask)