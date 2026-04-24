from afqa_toolbox.features import FeatRDI
from .helper import _finalize

def compute_rdi(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatRDI(blk_size=blk_size)
    rdi_map = feat.rdi(image,mask)

  

    return _finalize(rdi_map, mask)