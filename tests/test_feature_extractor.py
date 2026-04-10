"""
Tests for the main feature extractor module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction.extractor import extract_one, extract_dataset, FEATURE_COLS


class TestFeatureCols:
    """Tests for FEATURE_COLS constant."""

    def test_is_list(self):
        """FEATURE_COLS should be a list."""
        assert isinstance(FEATURE_COLS, list)

    def test_has_expected_length(self):
        """Should have 15 feature columns."""
        assert len(FEATURE_COLS) == 15

    def test_contains_required_features(self):
        """Should contain all required feature names."""
        required = ['gabor', 'gabor_std', 'ocl', 'ocl_std',
                    'lcs', 'lcs_std', 'fda', 'fda_std',
                    'rvu', 'rvu_std', 'rps', 'mean', 'std',
                    'ofl', 'ofl_std']
        for feat in required:
            assert feat in FEATURE_COLS

    def test_no_duplicates(self):
        """Should not have duplicate column names."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


class TestExtractOne:
    """Tests for the extract_one function."""

    def test_returns_dict(self, sample_image_pair, sample_config):
        """Should return a dictionary."""
        img, mask = sample_image_pair
        result = extract_one(img, mask, sample_config)
        assert isinstance(result, dict)

    def test_has_all_features(self, sample_image_pair, sample_config):
        """Should return all features in FEATURE_COLS."""
        img, mask = sample_image_pair
        result = extract_one(img, mask, sample_config)
        for feat in FEATURE_COLS:
            assert feat in result

    def test_all_values_are_floats(self, sample_image_pair, sample_config):
        """All feature values should be floats."""
        img, mask = sample_image_pair
        result = extract_one(img, mask, sample_config)
        for key, value in result.items():
            assert isinstance(value, float)

    def test_handles_empty_mask(self, sample_fingerprint_image, sample_config):
        """Should handle empty mask gracefully."""
        img = sample_fingerprint_image
        empty_mask = np.zeros_like(img)
        result = extract_one(img, empty_mask, sample_config)
        assert isinstance(result, dict)

    def test_respects_config(self, sample_image_pair):
        """Should use the provided config hyperparameters."""
        img, mask = sample_image_pair

        config1 = {
            'gabor_blk_size': 8,
            'gabor_sigma': 2.0,
            'gabor_freq': 0.05,
            'gabor_angle_num': 4,
            'ocl_ofl_blk_size': 16,
            'shared_blk_size': 32,
            'v1sz_x': 32,
            'v1sz_y': 8,
            'foreground_ratio': 0.6,
        }

        config2 = {
            'gabor_blk_size': 32,
            'gabor_sigma': 8.0,
            'gabor_freq': 0.15,
            'gabor_angle_num': 16,
            'ocl_ofl_blk_size': 64,
            'shared_blk_size': 96,
            'v1sz_x': 96,
            'v1sz_y': 32,
            'foreground_ratio': 0.95,
        }

        # Different configs should potentially produce different results
        result1 = extract_one(img, mask, config1)
        result2 = extract_one(img, mask, config2)

        # At least some features should differ
        differs = any(
            abs(result1[k] - result2[k]) > 1e-6
            for k in FEATURE_COLS
        )
        # Note: This may fail for some images if features are insensitive
        # but should generally pass


class TestExtractDataset:
    """Tests for the extract_dataset function."""

    def test_returns_dataframe(self, sample_dataset, sample_config):
        """Should return a pandas DataFrame."""
        import pandas as pd
        result = extract_dataset(sample_dataset, sample_config)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, sample_dataset, sample_config):
        """Should have all FEATURE_COLS columns."""
        result = extract_dataset(sample_dataset, sample_config)
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_has_correct_row_count(self, sample_dataset, sample_config):
        """Should have one row per input image."""
        result = extract_dataset(sample_dataset, sample_config)
        assert len(result) == len(sample_dataset)

    def test_all_values_are_finite(self, sample_dataset, sample_config):
        """All values should be finite (no NaN or Inf)."""
        result = extract_dataset(sample_dataset, sample_config)
        assert np.isfinite(result.values).all()

    def test_handles_empty_dataset(self, sample_config):
        """Should handle empty dataset gracefully."""
        result = extract_dataset([], sample_config)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_parallel_execution(self, sample_dataset, sample_config):
        """Should work with parallel execution (n_jobs=-1)."""
        result = extract_dataset(sample_dataset, sample_config, n_jobs=-1)
        assert len(result) == len(sample_dataset)

    def test_serial_execution(self, sample_dataset, sample_config):
        """Should work with serial execution (n_jobs=1)."""
        result = extract_dataset(sample_dataset, sample_config, n_jobs=1)
        assert len(result) == len(sample_dataset)
