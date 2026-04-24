import numpy as np
from afqa_toolbox.features import covcoef, orient, FeatFDA # type: ignore
from .helper import _finalize
# from your_library import FeatFDA

def compute_fda(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatFDA(blk_size=blk_size)
    fda_map = feat.fda(image,mask)

    

    return _finalize(fda_map, mask)