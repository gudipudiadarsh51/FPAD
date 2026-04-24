from afqa_toolbox.features import FeatACUT
from .helper import _finalize

def compute_acut(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatACUT(blk_size=blk_size)
    acut_map = feat.acut(image,mask)


    return _finalize(acut_map, mask)