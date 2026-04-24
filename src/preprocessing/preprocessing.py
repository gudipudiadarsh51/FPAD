"""
preprocessor.py
Loads raw fingerprint images (.png, .bmp, .wsq, .jpg, .tiff)
and generates (img, mask) pairs ready for feature extraction.

Adapt the mask import at the bottom to match your actual module name.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ── WSQ support (optional) ────────────────────────────────────────────────────

#connectedComponentWithStats

SUPPORTED_EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tiff', '.tif'}

import cv2
import numpy as np
from afqa_toolbox.features import FeatStats

import cv2
import numpy as np
from afqa_toolbox.features import FeatStats


import cv2
import numpy as np
from afqa_toolbox.features import FeatStats


def extract_foreground_afqa(
    img: np.ndarray,
    blk_size: int = 32,
    min_std: float = 13,
    min_size: int = 128,
):
    """
    Robust AFQA-compatible foreground extraction.

    Steps:
    1. Normalize
    2. AFQA std-based mask
    3. Morphological cleaning
    4. Largest component
    5. Crop foreground
    6. Enforce minimum size
    7. Pad to block-size multiple (CRITICAL)

    Returns:
        foreground (uint8), mask (uint8)
    """

    # ── Normalize ─────────────────────────────────────
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Local statistics (AFQA style) ─────────────────
    dummy_mask = np.ones_like(img, dtype=np.uint8)
    _, stds = FeatStats(blk_size).stats(img, dummy_mask)

    # ── Threshold std → foreground mask ───────────────
    _, mask_small = cv2.threshold(stds, min_std, 1, cv2.THRESH_BINARY)

    # ── Resize mask back ─────────────────────────────
    mask = cv2.resize(
        mask_small,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    # ── Morphological cleaning ───────────────────────
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ── Keep largest connected component ─────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # ── Apply mask ───────────────────────────────────
    foreground = img * mask

    # ── Crop to foreground bounding box ──────────────
    ys, xs = np.where(mask == 1)

    if len(ys) > 0 and len(xs) > 0:
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        foreground = foreground[ymin:ymax+1, xmin:xmax+1]
        mask = mask[ymin:ymax+1, xmin:xmax+1]

    # ── Ensure minimum size (prevents AFQA crashes) ──
    h, w = foreground.shape

    if h < min_size or w < min_size:
        new_size = max(min_size, max(h, w))

        foreground = cv2.resize(
            foreground,
            (new_size, new_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (new_size, new_size),
            interpolation=cv2.INTER_NEAREST
        )

    # ── Pad to multiple of block size (CRITICAL FIX) ─
    h, w = foreground.shape

    new_h = ((h + blk_size - 1) // blk_size) * blk_size
    new_w = ((w + blk_size - 1) // blk_size) * blk_size

    pad_h = new_h - h
    pad_w = new_w - w

    foreground = cv2.copyMakeBorder(
        foreground,
        0, pad_h,
        0, pad_w,
        cv2.BORDER_CONSTANT,
        value=0
    )

    mask = cv2.copyMakeBorder(
        mask,
        0, pad_h,
        0, pad_w,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return foreground, mask


def load_image(path: str) -> np.ndarray:
    """
    Load any supported fingerprint image format.
    Returns uint8 grayscale numpy array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    # Ensure uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def load_dataset(root_dir: str, mask_fn) -> Tuple[List[Tuple], np.ndarray, List[str]]:
    """
    Scans root_dir/live/ and root_dir/fake/ for fingerprint images.
    Calls mask_fn(img) on each image to generate the segmentation mask.

    Parameters
    ----------
    root_dir : str
        Path to dataset split folder. Must contain live/ and fake/ subfolders.
    mask_fn : callable
        Your mask generation function — called as mask_fn(img).
        Should return a binary uint8 numpy array same shape as img.

    Returns
    -------
    pairs  : list of (img, mask) tuples — numpy arrays
    labels : np.ndarray of int  (1 = live, 0 = fake)
    paths  : list of str — image file paths (for debugging)
    """
    root = Path(root_dir)
    pairs, labels, paths = [], [], []
    skipped = 0

    for label_name, label_val in [('live', 1), ('fake', 0)]:
        folder = root / label_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Structure must be: {root_dir}/live/ and {root_dir}/fake/"
            )

        img_paths = sorted(
            p for p in folder.glob('*')
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if len(img_paths) == 0:
            print(f"Warning: no images found in {folder}")

        for img_path in img_paths:
            try:
                img  = load_image(str(img_path))
                if  mask_fn is not None:
                    mask = mask_fn(img)
                else:
                    _,mask=extract_foreground_afqa(img)

                # Validate mask
                if mask.shape != img.shape:
                    raise ValueError(
                        f"Mask shape {mask.shape} != image shape {img.shape}"
                    )

                pairs.append((img, mask))
                labels.append(label_val)
                paths.append(str(img_path))

            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")
                skipped += 1

    if skipped > 0:
        print(f"Warning: skipped {skipped} images due to load errors")

    live_count = labels.count(1)
    fake_count = labels.count(0)
    print(f"Loaded from {root_dir}: {live_count} live, {fake_count} fake "
          f"({len(pairs)} total)")

    if live_count == 0 or fake_count == 0:
        raise ValueError(
            f"Need both classes — got live={live_count}, fake={fake_count}"
        )

    return pairs, np.array(labels, dtype=np.int64), paths


