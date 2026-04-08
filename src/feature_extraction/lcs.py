# src/feature_extraction/lcs.py

import numpy as np
import cv2
from afqa_toolbox.features import ( # type: ignore
    slanted_block_properties,
    covcoef,
    orient,FeatLCS
)

# Make sure FeatLCS is imported properly
# from your_library import FeatLCS


def compute_lcs(img: np.ndarray, mask: np.ndarray,
                config: dict) -> tuple[float, float]:

    rows, cols = img.shape

    blk_offset, map_rows, map_cols = slanted_block_properties(
        img.shape, config['blk_size'], config['v1sz_x'], config['v1sz_y']
    )

    result = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

    # Pad image and mask
    b_img = cv2.copyMakeBorder(
        img,
        blk_offset, blk_offset,
        blk_offset, blk_offset,
        cv2.BORDER_CONSTANT,
        value=0
    )

    b_mask = cv2.copyMakeBorder(
        mask,
        blk_offset, blk_offset,
        blk_offset, blk_offset,
        cv2.BORDER_CONSTANT,
        value=0
    )

    br = 0
    for r in range(blk_offset, (blk_offset + rows) - config['blk_size'] - 1, config['blk_size']):

        bc = 0
        for c in range(blk_offset, (blk_offset + cols) - config['blk_size'] - 1, config['blk_size']):

            patch = b_img[r:r+config['blk_size'], c:c+config['blk_size']]
            m     = b_mask[r:r+config['blk_size'], c:c+config['blk_size']]

            if m.mean() >= config['foreground_ratio']:

                cova, covb, covc = covcoef(patch, "c_diff_cv")
                theta = orient(cova, covb, covc)

                blkwim = b_img[
                    r-blk_offset:r+config['blk_size']+blk_offset,
                    c-blk_offset:c+config['blk_size']+blk_offset
                ]

                val_a = FeatLCS.lcs_block(
                    blkwim, theta,
                    config['v1sz_x'], config['v1sz_y'],
                    pad=False
                )

                val_b = FeatLCS.lcs_block(
                    blkwim, theta + np.pi/2.0,
                    config['v1sz_x'], config['v1sz_y'],
                    pad=False
                )

                # Keep best valid
                if np.isnan(val_a) and np.isnan(val_b):
                    val = np.nan
                elif np.isnan(val_a):
                    val = val_b
                elif np.isnan(val_b):
                    val = val_a
                else:
                    val = max(val_a, val_b)

                result[br, bc] = val

            bc += 1
        br += 1

    # Summary statistics
    lcs_mean = float(np.nanmean(result))
    lcs_std = float(np.nanstd(result))
    return lcs_mean, lcs_std
