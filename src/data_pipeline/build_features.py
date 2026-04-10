"""
build_features.py
Extract fingerprint features from live and fake (spoof) images.

Usage:
    python -m src.data_pipeline.build_features --data_dir data/train --output_csv data/train_features.csv
    python -m src.data_pipeline.build_features --data_dir data/val --output_csv data/val_features.csv

Dataset structure:
    data/
        live/
            image1.png
            image2.bmp
            ...
        fake/
            spoof1.png
            spoof2.jpg
            ...
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction.extractor import FeatureExtractor, FEATURE_COLS


SUPPORTED_EXTENSIONS = ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tiff", "*.tif")


def load_images_from_folder(folder: str):
    """Load all images from a folder with supported extensions."""
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
        image_paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(image_paths))


def extract_features_for_split(
    extractor: FeatureExtractor,
    folder: str,
    label: int,
    split_name: str,
) -> list:
    """
    Extract features for all images in a folder.

    Parameters
    ----------
    extractor : FeatureExtractor instance
    folder : path to folder containing images
    label : 1 for live, 0 for fake
    split_name : 'live' or 'fake' for logging

    Returns
    -------
    rows : list of dicts with filename, label, and feature columns
    """
    rows = []
    image_paths = load_images_from_folder(folder)

    if len(image_paths) == 0:
        print(f"Warning: No images found in {folder}")
        return rows

    print(f"Processing {split_name}: {len(image_paths)} images")

    for path in tqdm(image_paths, desc=f"Extracting {split_name}"):
        try:
            img = cv2.imread(path)

            if img is None:
                print(f"Skipping (could not read): {path}")
                continue

            # Extract features using the unified extractor
            features = extractor.extract(img)

            row = {
                "filename": os.path.basename(path),
                "label": label,
                "split": split_name,
            }

            # Add all features from FEATURE_COLS
            for feat in FEATURE_COLS:
                row[feat] = features.get(feat, 0.0)

            rows.append(row)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract fingerprint features from live and fake images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Root directory containing live/ and fake/ subfolders"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/train_features.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON for hyperparameters (from experiment.py)"
    )

    args = parser.parse_args()

    # Validate input directories
    data_dir = Path(args.data_dir)
    live_dir = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/Digital_Persona/data/live"
    fake_dir = "/Users/adarshgudipudi/Desktop/FPAD/FPAD/src/data/Digital_Persona/data/fake"

    if not live_dir.exists():
        raise FileNotFoundError(f"Live images folder not found: {live_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake images folder not found: {fake_dir}")

    # Initialize extractor with optional config
    print("Initializing FeatureExtractor...")
    config = {}
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")

    extractor = FeatureExtractor(config if config else None)

    # Extract features for both splits
    print("\n" + "=" * 50)
    rows_live = extract_features_for_split(
        extractor, str(live_dir), label=1, split_name="live"
    )
    rows_fake = extract_features_for_split(
        extractor, str(fake_dir), label=0, split_name="fake"
    )

    # Combine and save
    all_rows = rows_live + rows_fake

    if len(all_rows) == 0:
        print("Error: No features extracted. Check your data.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Report class distribution
    live_count = (df['label'] == 1).sum()
    fake_count = (df['label'] == 0).sum()
    print(f"\n{'=' * 50}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Live samples: {live_count}")
    print(f"Fake samples: {fake_count}")

    # Ensure all FEATURE_COLS are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Save to CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
