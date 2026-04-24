import os
import glob
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.preprocessing import extract_foreground_afqa
from src.feature_extraction.extractor import extract_dataset


import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "src/data/CrossMatch"
OUT_PATH = "data/features_CrossMatch.csv"


CONFIG = {
    'gabor_blk_size': 16,
    'gabor_sigma': 6.0,
    'gabor_freq': 0.1,
    'gabor_angle_num': 8,
    'ocl_ofl_blk_size': 32,
    'shared_blk_size': 32,
    'v1sz_x': 32,
    'v1sz_y': 16,
    'foreground_ratio': 0.8
}


def load_custom_dataset():

    pairs, labels, paths = [], [], []

    for folder, label in [("Live", 1), ("Fake", 0)]:
        folder_path = os.path.join(DATA_DIR, folder)

        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        print(f"{folder}: {len(image_paths)} images")

        for path in image_paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img, mask = extract_foreground_afqa(img)

                pairs.append((img, mask))
                labels.append(label)
                paths.append(path)

            except Exception as e:
                print(f"Skipping {path}: {e}")

    return pairs, np.array(labels), paths


def main():

    print("Loading dataset...")
    pairs, labels, paths = load_custom_dataset()

    print(f"\nTotal samples: {len(pairs)}")

    print("\nExtracting features...")
    df = extract_dataset(pairs, CONFIG)

    df["label"] = labels
    df["path"]  = paths

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved → {OUT_PATH}")
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
