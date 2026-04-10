"""
inference.py
Load trained model and predict on new fingerprint images.

Usage:
    python inference.py --image path/to/fingerprint.png
    python inference.py --image path/to/fingerprint.png --model configs/tabnet_final
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_model(model_path: str):
    """Load trained TabNet model."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    model = TabNetClassifier()
    model.load_model(model_path)
    return model


def load_scalers(scalers_path: str):
    """Load fitted scalers."""
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers['robust'], scalers['quantile']


def load_config(config_path: str):
    """Load feature extraction config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def extract_features(image_path: str, config: dict):
    """Extract features from a single image."""
    from preprocessing.preprocessing import extract_foreground
    from feature_extractor import extract_one

    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Preprocess
    foreground, mask = extract_foreground(img)

    # Extract features
    features = extract_one(foreground, mask, config)

    return features, foreground, mask


def predict(
    image_path: str,
    model,
    robust_scaler,
    quantile_scaler,
    config: dict,
) -> dict:
    """
    Run full inference pipeline.

    Returns
    -------
    result : dict
        Contains prediction, probability, and features
    """
    # Extract features
    features, foreground, mask = extract_features(image_path, config)

    # Convert to array in correct order
    from feature_extractor import FEATURE_COLS
    feature_names = list(features.keys())
    X = np.array([features[col] for col in FEATURE_COLS], dtype=np.float32).reshape(1, -1)

    # Scale
    X_scaled = robust_scaler.transform(X)
    X_scaled = quantile_scaler.transform(X_scaled)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]

    return {
        'image': str(image_path),
        'prediction': 'live' if prediction == 1 else 'fake',
        'probability': float(probability),
        'confidence': float(max(probability, 1 - probability)),
        'features': features,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Predict fingerprint liveness using trained model'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to fingerprint image'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='configs/tabnet_final',
        help='Path to trained TabNet model'
    )
    parser.add_argument(
        '--scalers', '-s',
        type=str,
        default='configs/scalers.pkl',
        help='Path to fitted scalers'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/best_config.json',
        help='Path to feature extraction config'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file (optional)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    scalers_path = Path(args.scalers)
    config_path = Path(args.config)
    image_path = Path(args.image)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run training first: python src/training/training.py")
        return

    if not scalers_path.exists():
        print(f"Error: Scalers not found at {scalers_path}")
        return

    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # Load model and scalers
    print(f"Loading model from {model_path}...")
    model = load_model(str(model_path))

    print(f"Loading scalers from {scalers_path}...")
    robust, qt = load_scalers(str(scalers_path))

    print(f"Loading config from {config_path}...")
    config = load_config(str(config_path))

    # Run inference
    print(f"\nProcessing image: {image_path}")
    result = predict(str(image_path), model, robust, qt, config)

    # Output results
    print("\n" + "=" * 50)
    print(f"Image:        {result['image']}")
    print(f"Prediction:   {result['prediction'].upper()}")
    print(f"Probability:  {result['probability']:.4f}")
    print(f"Confidence:   {result['confidence']:.2%}")
    print("=" * 50)

    if args.verbose:
        print("\nExtracted Features:")
        for name, value in result['features'].items():
            print(f"  {name:<20} {value:.6f}")

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
