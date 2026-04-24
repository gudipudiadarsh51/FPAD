from afqa_toolbox.features import FeatOCL, covcoef # type: ignore
import numpy as np
from .helper import _finalize

def compute_ocl(image, mask, **kwargs):
    blk_size = kwargs.get("ocl_ofl_blk_size", 32)

    feat = FeatOCL(blk_size=blk_size)
    ocl_map = feat.ocl(image,mask)

   
    return _finalize(ocl_map, mask)