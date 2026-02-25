from afqa_toolbox.features import FeatOCL, covcoef # type: ignore
import numpy as np


def compute_ocl(image, mask,
                blk_size=32,
                foreground_ratio=0.8):

    H, W = image.shape
    ocl_values = []

    for r in range(0, H - blk_size + 1, blk_size):
        for c in range(0, W - blk_size + 1, blk_size):

            block_mask = mask[r:r+blk_size, c:c+blk_size]

            if block_mask.mean() >= foreground_ratio:

                block = image[r:r+blk_size, c:c+blk_size]

                a, b, c_cov_val = covcoef(block, "c_diff_cv")
                val = FeatOCL.ocl_block(a, b, c_cov_val)

                ocl_values.append(val)

    if len(ocl_values) == 0:
        return 0.0, 0.0

    ocl_values = np.array(ocl_values, dtype=np.float64)

    return (
        float(np.nanmean(ocl_values)),
        float(np.nanstd(ocl_values))
    )
