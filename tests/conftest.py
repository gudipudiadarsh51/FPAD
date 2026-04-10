"""
Pytest configuration and shared fixtures for FPAD tests.
"""

import numpy as np
import pytest
import cv2


@pytest.fixture
def sample_fingerprint_image():
    """
    Create a synthetic fingerprint-like image for testing.
    Returns a grayscale image with ridge-like patterns.
    """
    # Create a 512x512 synthetic fingerprint pattern
    h, w = 512, 512
    img = np.zeros((h, w), dtype=np.uint8)

    # Generate sinusoidal ridge pattern
    y, x = np.ogrid[:h, :w]
    frequency = 0.05
    img = ((np.sin(2 * np.pi * frequency * x) + 1) * 127.5).astype(np.uint8)

    # Add some noise to make it realistic
    noise = np.random.normal(0, 10, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def sample_mask(sample_fingerprint_image):
    """
    Create a binary mask for the sample fingerprint image.
    """
    img = sample_fingerprint_image
    mask = np.zeros_like(img)
    # Mark center region as foreground (simulating fingerprint area)
    h, w = img.shape
    mask[h//4:3*h//4, w//4:3*w//4] = 255
    return mask


@pytest.fixture
def sample_image_pair(sample_fingerprint_image, sample_mask):
    """
    Return a tuple of (image, mask) for testing.
    """
    return (sample_fingerprint_image, sample_mask)


@pytest.fixture
def sample_dataset(sample_image_pair):
    """
    Create a small dataset for testing.
    Returns list of (img, mask) tuples.
    """
    # Return 5 copies of the same image for testing
    return [sample_image_pair for _ in range(5)]


@pytest.fixture
def sample_config():
    """
    Default hyperparameter config for testing.
    """
    return {
        'gabor_blk_size': 16,
        'gabor_sigma': 6.0,
        'gabor_freq': 0.1,
        'gabor_angle_num': 8,
        'ocl_ofl_blk_size': 32,
        'shared_blk_size': 64,
        'v1sz_x': 64,
        'v1sz_y': 16,
        'foreground_ratio': 0.8,
    }


@pytest.fixture
def sample_labels():
    """
    Binary labels for testing (1=live, 0=fake).
    """
    return np.array([1, 1, 0, 0, 1], dtype=np.int64)


@pytest.fixture
def sample_feature_df():
    """
    Sample DataFrame with raw features for testing feature engineering.
    """
    import pandas as pd

    data = {
        'gabor': [0.5, 0.6, 0.4, 0.55, 0.45],
        'gabor_std': [0.1, 0.12, 0.08, 0.11, 0.09],
        'ocl': [0.7, 0.65, 0.75, 0.68, 0.72],
        'ocl_std': [0.15, 0.14, 0.16, 0.13, 0.17],
        'lcs': [0.3, 0.35, 0.28, 0.32, 0.29],
        'lcs_std': [0.05, 0.06, 0.04, 0.055, 0.045],
        'fda': [0.6, 0.55, 0.65, 0.58, 0.62],
        'fda_std': [0.1, 0.09, 0.11, 0.095, 0.105],
        'rvu': [0.4, 0.38, 0.42, 0.39, 0.41],
        'rvu_std': [0.08, 0.07, 0.09, 0.075, 0.085],
        'rps': [0.8, 0.75, 0.85, 0.78, 0.82],
        'mean': [128.0, 125.0, 130.0, 127.0, 129.0],
        'std': [50.0, 48.0, 52.0, 49.0, 51.0],
        'ofl': [0.55, 0.52, 0.58, 0.54, 0.56],
        'ofl_std': [0.12, 0.11, 0.13, 0.115, 0.125],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide a temporary directory for test outputs.
    """
    return tmp_path
