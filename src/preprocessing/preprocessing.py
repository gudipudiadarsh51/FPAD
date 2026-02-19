import cv2
import numpy as np


def extract_foreground(gray):
    """
    Extract and crop fingerprint foreground using Otsu thresholding
    and morphological cleaning.

    Parameters
    ----------
    gray : np.ndarray (H, W)
        Grayscale image

    Returns
    -------
    foreground_cropped : np.ndarray
    mask_cropped : np.ndarray
    """

    # ----------------------------------------------------------
    # 1) Ensure uint8
    # ----------------------------------------------------------
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # ----------------------------------------------------------
    # 2) Otsu threshold
    # ----------------------------------------------------------
    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure ridges are foreground
    if np.mean(gray[th == 255]) > np.mean(gray[th == 0]):
        th = cv2.bitwise_not(th)

    mask = (th > 0).astype(np.uint8)

    # ----------------------------------------------------------
    # 3) Morphological cleaning
    # ----------------------------------------------------------
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ----------------------------------------------------------
    # 4) Keep largest connected component
    # ----------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)

    # ----------------------------------------------------------
    # 5) Apply mask
    # ----------------------------------------------------------
    foreground = gray * mask

    # ----------------------------------------------------------
    # 6) Tight crop
    # ----------------------------------------------------------
    ys, xs = np.where(mask == 1)

    if len(ys) == 0 or len(xs) == 0:
        # No foreground detected
        return gray, mask

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    foreground_cropped = foreground[ymin:ymax+1, xmin:xmax+1]
    mask_cropped = mask[ymin:ymax+1, xmin:xmax+1]

    return foreground_cropped, mask_cropped
