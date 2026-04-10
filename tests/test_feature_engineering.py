"""
Tests for feature engineering module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering import engineer_fingerprint_features, get_feature_names


class TestEngineerFingerprintFeatures:
    """Tests for the engineer_fingerprint_features function."""

    def test_returns_dataframe(self, sample_feature_df):
        """Should return a pandas DataFrame."""
        import pandas as pd
        result = engineer_fingerprint_features(sample_feature_df)
        assert isinstance(result, pd.DataFrame)

    def test_increases_feature_count(self, sample_feature_df):
        """Should create more features than input."""
        original_cols = len(sample_feature_df.columns)
        result = engineer_fingerprint_features(sample_feature_df)
        assert len(result.columns) > original_cols

    def test_preserves_original_columns(self, sample_feature_df):
        """Should preserve all original columns."""
        original_cols = set(sample_feature_df.columns)
        result = engineer_fingerprint_features(sample_feature_df)
        assert original_cols.issubset(set(result.columns))

    def test_creates_snr_features(self, sample_feature_df):
        """Should create signal-to-noise ratio features."""
        result = engineer_fingerprint_features(sample_feature_df)
        snr_cols = [c for c in result.columns if '_snr' in c]
        assert len(snr_cols) >= 6  # At least 6 SNR features

    def test_creates_interaction_features(self, sample_feature_df):
        """Should create cross-feature interaction columns."""
        result = engineer_fingerprint_features(sample_feature_df)
        interaction_cols = [c for c in result.columns if '_x_' in c]
        assert len(interaction_cols) >= 5  # At least 5 interaction features

    def test_creates_composite_features(self, sample_feature_df):
        """Should create domain-specific composite features."""
        result = engineer_fingerprint_features(sample_feature_df)
        assert 'ridge_contrast' in result.columns
        assert 'orient_coherence' in result.columns
        assert 'texture_strength' in result.columns

    def test_creates_aggregate_features(self, sample_feature_df):
        """Should create aggregate statistics."""
        result = engineer_fingerprint_features(sample_feature_df)
        assert 'texture_mean' in result.columns
        assert 'texture_std' in result.columns
        assert 'quality_mean' in result.columns
        assert 'variability_mean' in result.columns

    def test_creates_difference_features(self, sample_feature_df):
        """Should create difference features."""
        result = engineer_fingerprint_features(sample_feature_df)
        assert 'gabor_ocl_diff' in result.columns
        assert 'lcs_fda_diff' in result.columns
        assert 'ofl_rvu_diff' in result.columns

    def test_creates_squared_features(self, sample_feature_df):
        """Should create squared term features."""
        result = engineer_fingerprint_features(sample_feature_df)
        squared_cols = [c for c in result.columns if '_sq' in c]
        assert len(squared_cols) >= 6  # At least 6 squared features

    def test_handles_nan_values(self):
        """Should handle NaN values gracefully."""
        import pandas as pd

        df_with_nans = pd.DataFrame({
            'gabor': [0.5, np.nan, 0.4],
            'gabor_std': [0.1, 0.12, np.nan],
            'ocl': [0.7, 0.65, 0.75],
            'ocl_std': [0.15, 0.14, 0.16],
            'lcs': [0.3, 0.35, 0.28],
            'lcs_std': [0.05, 0.06, 0.04],
            'fda': [0.6, 0.55, 0.65],
            'fda_std': [0.1, 0.09, 0.11],
            'rvu': [0.4, 0.38, 0.42],
            'rvu_std': [0.08, 0.07, 0.09],
            'rps': [0.8, 0.75, 0.85],
            'mean': [128.0, 125.0, 130.0],
            'std': [50.0, 48.0, 52.0],
            'ofl': [0.55, 0.52, 0.58],
            'ofl_std': [0.12, 0.11, 0.13],
        })

        result = engineer_fingerprint_features(df_with_nans)
        # Should not have NaN values after engineering
        assert result.isnull().sum().sum() == 0

    def test_handles_inf_values(self):
        """Should handle Inf values gracefully."""
        import pandas as pd

        df_with_inf = pd.DataFrame({
            'gabor': [0.5, np.inf, 0.4],
            'gabor_std': [0.1, 0.12, 0.08],
            'ocl': [0.7, 0.65, 0.75],
            'ocl_std': [0.15, 0.14, 0.16],
            'lcs': [0.3, 0.35, 0.28],
            'lcs_std': [0.05, 0.06, 0.04],
            'fda': [0.6, 0.55, 0.65],
            'fda_std': [0.1, 0.09, 0.11],
            'rvu': [0.4, 0.38, 0.42],
            'rvu_std': [0.08, 0.07, 0.09],
            'rps': [0.8, 0.75, 0.85],
            'mean': [128.0, 125.0, 130.0],
            'std': [50.0, 48.0, 52.0],
            'ofl': [0.55, 0.52, 0.58],
            'ofl_std': [0.12, 0.11, 0.13],
        })

        result = engineer_fingerprint_features(df_with_inf)
        # Should not have Inf values after engineering
        assert not np.isinf(result.values).any()

    def test_same_row_count(self, sample_feature_df):
        """Should preserve the number of rows."""
        original_rows = len(sample_feature_df)
        result = engineer_fingerprint_features(sample_feature_df)
        assert len(result) == original_rows

    def test_no_duplicate_columns(self, sample_feature_df):
        """Should not create duplicate column names."""
        result = engineer_fingerprint_features(sample_feature_df)
        assert len(result.columns) == len(set(result.columns))


class TestGetFeatureNames:
    """Tests for the get_feature_names function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_feature_names()
        assert isinstance(result, list)

    def test_contains_base_features(self):
        """Should include all base feature names."""
        result = get_feature_names()
        base_features = ['gabor', 'gabor_std', 'ocl', 'ocl_std',
                         'lcs', 'lcs_std', 'fda', 'fda_std']
        for feat in base_features:
            assert feat in result

    def test_contains_engineered_features(self):
        """Should include engineered feature names."""
        result = get_feature_names()
        assert 'gabor_snr' in result
        assert 'gabor_x_fda' in result
        assert 'ridge_contrast' in result
        assert 'texture_mean' in result
        assert 'gabor_sq' in result
