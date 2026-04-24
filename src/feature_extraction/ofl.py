import numpy as np
from afqa_toolbox.features import covcoef, orient, FeatOFL # type: ignore
from .helper import _finalize
# Make sure FeatOFL is imported correctly
# from your_library import FeatOFL


def compute_ofl(image, mask, **kwargs):
    blk_size = kwargs.get("ocl_ofl_blk_size", 32)
    v1sz_x   = kwargs.get("v1sz_x", blk_size)
    v1sz_y   = kwargs.get("v1sz_y", 16)
    fg_ratio = kwargs.get("foreground_ratio", 0.8)

    if mask is None:
        mask = np.ones_like(image, dtype=np.uint8)

    feat = FeatOFL(blk_size=blk_size, v1sz_x=v1sz_x, v1sz_y=v1sz_y, foreground_ratio=fg_ratio)
    ofl_map = feat.ofl(image, mask)

    return _finalize(ofl_map, mask)