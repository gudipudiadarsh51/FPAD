import numpy as np
from afqa_toolbox.features import covcoef, orient, FeatRVU # type: ignore
from .helper import _finalize
# from your_library import FeatRVU


def compute_rvu(img, mask, **kwargs):

    blk_size = kwargs.get('shared_blk_size', 32)
    v1sz_x = min(kwargs.get('v1sz_x', blk_size), blk_size)
    v1sz_y = min(kwargs.get('v1sz_y', blk_size // 2), blk_size)
    foreground_ratio = kwargs.get('foreground_ratio', 0.8)

    if mask is None:
        mask=np.ones_like(img, dtype=np.uint8)
    
    feat = FeatRVU(blk_size=blk_size, v1sz_x=v1sz_x, v1sz_y=v1sz_y,foreground_ratio=foreground_ratio)  # Initialize the feature extractor
    rvu_map = feat.rvu(img, mask)  # Compute the RVU standard deviation map

   
    
    return _finalize(rvu_map, mask)
    
