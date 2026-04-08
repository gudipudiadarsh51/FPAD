import numpy as np
from afqa_toolbox.features import covcoef, orient, FeatFDA # type: ignore

# from your_library import FeatFDA


def compute_fda(img, mask, config: dict) -> tuple[float, float]:

    H, W = img.shape
    rows = H // config['blk_size']
    cols = W // config['blk_size']

    fda_map = np.full((rows, cols), np.nan, dtype=np.float64)
    vals = []

    br = 0
    for r in range(0, H - config['blk_size'] + 1, config['blk_size']):

        bc = 0
        for c in range(0, W - config['blk_size'] + 1, config['blk_size']):

            block_mask = mask[r:r+config['blk_size'], c:c+config['blk_size']]

            if block_mask.mean() >= config['foreground_ratio']:

                block = img[r:r+config['blk_size'], c:c+config['blk_size']]

                a, b, c_cov_val = covcoef(block, "c_diff_cv")
                theta = orient(a, b, c_cov_val)

                val = FeatFDA.fda_block(
                    block,
                    theta,
                    config['v1sz_x'],
                    config['v1sz_y'],
                    pad=False
                )

                fda_map[br, bc] = val
                vals.append(val)

            bc += 1
        br += 1

    if len(vals) == 0:
        return 0.0, 0.0

    vals = np.array(vals, dtype=np.float64)

    return (
        float(np.nanmean(vals)),
        float(np.nanstd(vals))
    )
