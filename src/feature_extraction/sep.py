from afqa_toolbox.features import FeatSEP
from .helper import _finalize

def compute_sep(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatSEP(blk_size=blk_size)
    sep_map = feat.sep(image, mask)



    return _finalize(sep_map, mask)