"""
Tests for preprocessing module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocessing import extract_foreground


class TestExtractForeground:
    """Tests for the extract_foreground function."""

    def test_returns_tuple(self, sample_fingerprint_image):
        """Should return a tuple of (foreground, mask)."""
        result = extract_foreground(sample_fingerprint_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_foreground_is_numpy_array(self, sample_fingerprint_image):
        """Foreground should be a numpy array."""
        foreground, mask = extract_foreground(sample_fingerprint_image)
        assert isinstance(foreground, np.ndarray)
        assert isinstance(mask, np.ndarray)

    def test_mask_is_binary(self, sample_fingerprint_image):
        """Mask should contain only 0 and 1 values."""
        _, mask = extract_foreground(sample_fingerprint_image)
        unique_values = np.unique(mask)
        assert all(v in [0, 1] for v in unique_values)

    def test_mask_same_shape_as_input(self, sample_fingerprint_image):
        """Mask should have the same shape as input image."""
        _, mask = extract_foreground(sample_fingerprint_image)
        assert mask.shape == sample_fingerprint_image.shape

    def test_handles_empty_image(self):
        """Should handle an all-zeros image gracefully."""
        empty_img = np.zeros((100, 100), dtype=np.uint8)
        foreground, mask = extract_foreground(empty_img)
        assert foreground is not None
        assert mask is not None

    def test_handles_uint8_input(self, sample_fingerprint_image):
        """Should handle uint8 input correctly."""
        assert sample_fingerprint_image.dtype == np.uint8
        foreground, mask = extract_foreground(sample_fingerprint_image)
        assert foreground is not None

    def test_cropped_foreground_not_empty(self, sample_fingerprint_image):
        """Cropped foreground should not be empty for valid images."""
        foreground, mask = extract_foreground(sample_fingerprint_image)
        assert foreground.size > 0

    def test_mask_has_foreground_region(self, sample_fingerprint_image):
        """Mask should have at least some foreground pixels for valid images."""
        _, mask = extract_foreground(sample_fingerprint_image)
        assert np.sum(mask > 0) > 0

    def test_grayscale_input(self, sample_fingerprint_image):
        """Should work with grayscale input."""
        assert len(sample_fingerprint_image.shape) == 2
        foreground, mask = extract_foreground(sample_fingerprint_image)
        assert foreground is not None

    def test_small_image(self):
        """Should handle small images."""
        small_img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        foreground, mask = extract_foreground(small_img)
        assert foreground is not None
        assert mask is not None

    def test_large_image(self):
        """Should handle large images."""
        large_img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        foreground, mask = extract_foreground(large_img)
        assert foreground is not None
        assert mask is not None
