from afqa_toolbox.features import FeatOCL, covcoef # type: ignore
import numpy as np


def compute_ocl(image: np.ndarray, mask: np.ndarray,
                config: dict) -> tuple[float, float]:

    H, W = image.shape
    ocl_values = []

    for r in range(0, H - config['blk_size'] + 1, config['blk_size']):
        for c in range(0, W - config['blk_size'] + 1, config['blk_size']):

            block_mask = mask[r:r+config['blk_size'], c:c+config['blk_size']]

            if block_mask.mean() >= config['foreground_ratio']:

                block = image[r:r+config['blk_size'], c:c+config['blk_size']]

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
