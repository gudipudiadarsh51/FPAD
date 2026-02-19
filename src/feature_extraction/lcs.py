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


def compute_lcs(img, mask,
                blk_size=64,
                v1sz_x=64,
                v1sz_y=16,
                foreground_ratio=0.8):

    rows, cols = img.shape

    blk_offset, map_rows, map_cols = slanted_block_properties(
        img.shape, blk_size, v1sz_x, v1sz_y
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
    for r in range(blk_offset, (blk_offset + rows) - blk_size - 1, blk_size):

        bc = 0
        for c in range(blk_offset, (blk_offset + cols) - blk_size - 1, blk_size):

            patch = b_img[r:r+blk_size, c:c+blk_size]
            m     = b_mask[r:r+blk_size, c:c+blk_size]

            if m.mean() >= foreground_ratio:

                cova, covb, covc = covcoef(patch, "c_diff_cv")
                theta = orient(cova, covb, covc)

                blkwim = b_img[
                    r-blk_offset:r+blk_size+blk_offset,
                    c-blk_offset:c+blk_size+blk_offset
                ]

                val_a = FeatLCS.lcs_block(
                    blkwim, theta,
                    v1sz_x, v1sz_y,
                    pad=False
                )

                val_b = FeatLCS.lcs_block(
                    blkwim, theta + np.pi/2.0,
                    v1sz_x, v1sz_y,
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

    return lcs_mean
