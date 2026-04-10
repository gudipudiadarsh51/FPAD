"""
Tests for feature extraction modules.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.gabor import compute_gabor
from feature_extraction.fda import compute_fda
from feature_extraction.lcs import compute_lcs
from feature_extraction.ocl import compute_ocl
from feature_extraction.ofl import compute_ofl
from feature_extraction.rvu import compute_rvu
from feature_extraction.rps import compute_rps
from feature_extraction.mean import compute_mean
from feature_extraction.std import compute_std


class TestComputeGabor:
    """Tests for Gabor feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_gabor(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_gabor(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_handles_empty_mask(self, sample_fingerprint_image):
        """Should handle empty mask gracefully."""
        img = sample_fingerprint_image
        empty_mask = np.zeros_like(img)
        mean, std = compute_gabor(img, empty_mask)
        assert mean == 0.0
        assert std == 0.0

    def test_non_negative_std(self, sample_image_pair):
        """Standard deviation should be non-negative."""
        img, mask = sample_image_pair
        _, std = compute_gabor(img, mask)
        assert std >= 0


class TestComputeFDA:
    """Tests for FDA feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_fda(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_fda(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_handles_empty_mask(self, sample_fingerprint_image):
        """Should handle empty mask gracefully."""
        img = sample_fingerprint_image
        empty_mask = np.zeros_like(img)
        mean, std = compute_fda(img, empty_mask)
        assert mean == 0.0
        assert std == 0.0


class TestComputeLCS:
    """Tests for LCS feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_lcs(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_lcs(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestComputeOCL:
    """Tests for OCL feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_ocl(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_ocl(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestComputeOFL:
    """Tests for OFL feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_ofl(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_ofl(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestComputeRVU:
    """Tests for RVU feature computation."""

    def test_returns_tuple(self, sample_image_pair):
        """Should return a tuple of (mean, std)."""
        img, mask = sample_image_pair
        result = compute_rvu(img, mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_image_pair):
        """Should return float values."""
        img, mask = sample_image_pair
        mean, std = compute_rvu(img, mask)
        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestComputeRPS:
    """Tests for RPS feature computation."""

    def test_returns_tuple(self, sample_fingerprint_image):
        """Should return a tuple of (mean, std)."""
        img = sample_fingerprint_image
        result = compute_rps(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self, sample_fingerprint_image):
        """Should return float values."""
        img = sample_fingerprint_image
        mean, std = compute_rps(img)
        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestComputeMean:
    """Tests for mean intensity computation."""

    def test_returns_float(self, sample_image_pair):
        """Should return a float value."""
        img, mask = sample_image_pair
        result = compute_mean(img, mask)
        assert isinstance(result, float)

    def test_within_valid_range(self, sample_image_pair):
        """Mean should be in valid range [0, 255]."""
        img, mask = sample_image_pair
        mean = compute_mean(img, mask)
        assert 0 <= mean <= 255

    def test_handles_empty_mask(self, sample_fingerprint_image):
        """Should handle empty mask gracefully."""
        img = sample_fingerprint_image
        empty_mask = np.zeros_like(img)
        mean = compute_mean(img, empty_mask)
        assert isinstance(mean, float)


class TestComputeStd:
    """Tests for standard deviation computation."""

    def test_returns_float(self, sample_image_pair):
        """Should return a float value."""
        img, mask = sample_image_pair
        result = compute_std(img, mask)
        assert isinstance(result, float)

    def test_non_negative(self, sample_image_pair):
        """Standard deviation should be non-negative."""
        img, mask = sample_image_pair
        std = compute_std(img, mask)
        assert std >= 0


class TestFeatureSignatures:
    """Tests for consistent function signatures across all feature modules."""

    def test_all_features_accept_img_mask_kwargs(self, sample_image_pair):
        """All feature functions should accept (img, mask, **kwargs)."""
        img, mask = sample_image_pair

        functions = [
            compute_gabor, compute_fda, compute_lcs,
            compute_ocl, compute_ofl, compute_rvu,
            compute_mean, compute_std
        ]

        for func in functions:
            # Should not raise TypeError for unexpected keyword arguments
            try:
                result = func(img, mask, some_extra_param="test")
                assert result is not None
            except TypeError as e:
                pytest.fail(f"{func.__name__} does not accept **kwargs: {e}")
