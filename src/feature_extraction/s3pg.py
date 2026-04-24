from afqa_toolbox.features import FeatS3PG
from .helper import _finalize

def compute_s3pg(image, mask, **kwargs):
    blk_size = kwargs.get("shared_blk_size", 32)

    feat = FeatS3PG(blk_size=blk_size)
    s3pg_map = feat.s3pg(image, mask)

    

    return _finalize(s3pg_map, mask)