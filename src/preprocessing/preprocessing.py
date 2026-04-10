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

def extract_foreground(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Extract the fingerprint foreground from the background using otsu thresholding and 
    morpholigical cleaning.'''
     
     #ensure greyscale
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Otsu thresholding
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure ridges are foreground (not background)
    if np.mean(img[th == 255]) > np.mean(img[th == 0]):
        th = cv2.bitwise_not(th)

    mask= (th>0).astype(np.uint8)

    #mopholigical cleaning
    kernel=np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    #Keep largest connected component
    num_labels, labels, stats,_=cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels>1:
        largest_label = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask=(labels ==largest_label).astype(np.uint8)

    #apply mask to image
    foreground = img*mask

    #tight crop
    ys,xs=np.where(mask==1)

    if len(ys) ==0 or len(xs) ==0:
        #the foreground is detected
        return img, mask
    
    

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
                    _,mask=extract_foreground(img)

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


