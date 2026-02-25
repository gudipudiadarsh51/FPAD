import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_extraction.extractor import FeatureExtractor


IMAGE_DIR = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/images/Digital_Persona/Live"
OUTPUT_CSV = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/features.csv"


def main():

    extractor = FeatureExtractor()

    rows = []

    image_paths = []
    for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    print("Total Live images found:", len(image_paths))

    for path in tqdm(image_paths, desc="Processing Live"):

        img = cv2.imread(path)

        if img is None:
            print("Skipping:", path)
            continue

        features = extractor.extract(img)

        row = {
            "filename": os.path.basename(path),
            "label": "Live"
        }

        for i, value in enumerate(features):
            row[f"feature_{i+1}"] = float(value)

        rows.append(row)

    df = pd.DataFrame(rows)

    print("Final dataset shape:", df.shape)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved to:", OUTPUT_CSV)
    


if __name__ == "__main__":
    main()
