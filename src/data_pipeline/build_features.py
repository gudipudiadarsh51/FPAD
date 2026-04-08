import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.feature_extraction.extractor import FeatureExtractor
from src.experiment import build_cofig

BASE_DIR = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/images/Digital_Persona"
OUTPUT_CSV = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/features.csv"


def main():
    extractor = FeatureExtractor()
    rows = []
    config = build_cofig()  # Use default config for feature extraction. You can modify this to use specific parameters.    

    # Define folders + labels
    data_sources = {
        "Live": os.path.join(BASE_DIR, "Live"),
        "Fake": os.path.join(BASE_DIR, "Fake"),
    }

    for label, folder in data_sources.items():

        image_paths = []
        for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
            image_paths.extend(glob.glob(os.path.join(folder, ext)))

        print(f"Total {label} images found:", len(image_paths))

        for path in tqdm(image_paths, desc=f"Processing {label}"):

            img = cv2.imread(path)

            if img is None:
                print("Skipping:", path)
                continue

            features = extractor.extract(img,config)

            row = {
                "filename": os.path.basename(path),
                "label": label  
            }

            for i, value in enumerate(features):
                row[f"feature_{i+1}"] = float(value)

            rows.append(row)

    df = pd.DataFrame(rows)

    print("Final dataset shape:", df.shape)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved to:", OUTPUT_CSV)

    X_train,X_val,y_train,y_val = train_test_split(df.drop(columns=['filename','label']), df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    X_train.to_csv("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/features_train.csv", index=False)
    X_val.to_csv("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/features_val.csv", index=False)
    y_train.to_csv("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/labels_train.csv", index=False)
    y_val.to_csv("/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/labels_val.csv", index=False)
    print("Train/Val split saved.")
    


if __name__ == "__main__":
    main()