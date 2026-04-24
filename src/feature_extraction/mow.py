from afqa_toolbox.features import FeatMOW
from .helper import _finalize

def compute_mow(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatMOW(blk_size=blk_size)
    mow_map = feat.mow(image,mask)

  

    return _finalize(mow_map, mask)