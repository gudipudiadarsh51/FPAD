import numpy as np
from afqa_toolbox.features import covcoef, orient, FeatOFL # type: ignore

# Make sure FeatOFL is imported correctly
# from your_library import FeatOFL


def compute_ofl(img, mask,
                blk_size=32,
                foreground_ratio=0.8):

    H, W = img.shape

    rows = H // blk_size
    cols = W // blk_size

    orientation_map = np.full((rows, cols), np.nan)

    br = 0
    for r in range(0, H - blk_size + 1, blk_size):

        bc = 0
        for c in range(0, W - blk_size + 1, blk_size):

            block_mask = mask[r:r+blk_size, c:c+blk_size]

            if block_mask.mean() >= foreground_ratio:

                block = img[r:r+blk_size, c:c+blk_size]
                a, b, c_cov_val = covcoef(block, "c_diff_cv")
                orientation_map[br, bc] = orient(a, b, c_cov_val)

            bc += 1
        br += 1

    # Compute OFL map
    ofl_map = FeatOFL.ofl_blocks(
        orientation_map,
        ang_min_deg=0
    )

    # Safety check
    if np.all(np.isnan(ofl_map)):
        return 0.0, 0.0

    return (
        float(np.nanmean(ofl_map)),
        float(np.nanstd(ofl_map))
    )
