"""
mask.py - Mask generation for fingerprint images.

This module provides the generate_mask() function used by experiment.py
to create binary foreground masks from fingerprint images.
"""

import cv2
import numpy as np


def generate_mask(img: np.ndarray) -> np.ndarray:
    """
    Generate a binary foreground mask from a fingerprint image.

    Uses Otsu thresholding + morphological operations to segment
    the fingerprint foreground from the background.

    Parameters
    ----------
    img : np.ndarray
        Grayscale fingerprint image (uint8)

    Returns
    -------
    mask : np.ndarray
        Binary mask (uint8) with same shape as input.
        1 = foreground (fingerprint), 0 = background
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure ridges are foreground (not background)
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # Convert to binary mask
    mask = (thresh > 0).astype(np.uint8)

    # Morphological cleaning
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component (the fingerprint)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)

    return mask
