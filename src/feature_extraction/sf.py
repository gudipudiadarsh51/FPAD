from afqa_toolbox.features import FeatSF
from .helper import _finalize

def compute_sf(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatSF(blk_size=blk_size)
    sf_map = feat.sf(image, mask)

   

    return _finalize(sf_map, mask)