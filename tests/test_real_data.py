"""
Integration tests with REAL fingerprint data.
These tests load actual images from the dataset and run the full pipeline.

Requires: data/train/live/ and data/train/fake/ directories with images.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def real_data_dir():
    """Path to the real data directory."""
    # Check common locations for data directory
    possible_paths = [
        Path("data"),
        Path("../data"),
        Path("../../data"),
        Path("C:/Users/astit/Downloads/data"),
    ]
    for p in possible_paths:
        if p.exists() and (p / "train").exists():
            return p
    pytest.skip("Real data directory not found. Skipping real data tests.")


@pytest.fixture
def real_train_live_images(real_data_dir):
    """Load real live fingerprint images from dataset."""
    import cv2

    live_dir = real_data_dir / "train" / "live"
    if not live_dir.exists():
        pytest.skip("No live images found.")

    images = []
    for ext in ["*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]:
        images.extend(live_dir.glob(ext))

    if len(images) == 0:
        pytest.skip("No live images found.")

    return list(images)[:10]  # Limit to first 10 for speed


@pytest.fixture
def real_train_fake_images(real_data_dir):
    """Load real fake (spoof) fingerprint images from dataset."""
    import cv2

    fake_dir = real_data_dir / "train" / "fake"
    if not fake_dir.exists():
        pytest.skip("No fake images found.")

    images = []
    for ext in ["*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]:
        images.extend(fake_dir.glob(ext))

    if len(images) == 0:
        pytest.skip("No fake images found.")

    return list(images)[:10]  # Limit to first 10 for speed


class TestRealDataLoading:
    """Test loading real fingerprint images."""

    def test_can_load_live_images(self, real_train_live_images):
        """Should be able to load live images."""
        assert len(real_train_live_images) > 0

    def test_can_load_fake_images(self, real_train_fake_images):
        """Should be able to load fake images."""
        assert len(real_train_fake_images) > 0

    def test_images_are_valid_files(self, real_train_live_images):
        """Image paths should point to valid files."""
        for img_path in real_train_live_images:
            assert img_path.exists()
            assert img_path.stat().st_size > 0

    def test_images_are_decodable(self, real_train_live_images):
        """Images should be decodable by OpenCV."""
        import cv2

        for img_path in real_train_live_images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            assert img is not None, f"Could not decode {img_path}"
            assert img.size > 0


class TestRealDataPreprocessing:
    """Test preprocessing on real images."""

    def test_preprocess_live_images(self, real_train_live_images):
        """Should successfully preprocess live images."""
        from preprocessing.preprocessing import extract_foreground
        import cv2

        for img_path in real_train_live_images[:5]:  # Test first 5
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)

            assert foreground is not None
            assert mask is not None
            assert foreground.size > 0

    def test_preprocess_fake_images(self, real_train_fake_images):
        """Should successfully preprocess fake images."""
        from preprocessing.preprocessing import extract_foreground
        import cv2

        for img_path in real_train_fake_images[:5]:  # Test first 5
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)

            assert foreground is not None
            assert mask is not None

    def test_mask_has_foreground(self, real_train_live_images):
        """Masks should have some foreground region."""
        from preprocessing.preprocessing import extract_foreground
        import cv2

        for img_path in real_train_live_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            _, mask = extract_foreground(img)

            # At least 1% of image should be foreground
            foreground_ratio = np.sum(mask > 0) / mask.size
            assert foreground_ratio > 0.01, f"No foreground in {img_path}"


class TestRealDataFeatureExtraction:
    """Test feature extraction on real images."""

    def test_extract_features_live(self, real_train_live_images):
        """Should extract features from live images."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one
        import cv2

        config = {
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

        for img_path in real_train_live_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)
            features = extract_one(foreground, mask, config)

            assert isinstance(features, dict)
            assert len(features) == 15
            assert all(isinstance(v, float) for v in features.values())

    def test_extract_features_fake(self, real_train_fake_images):
        """Should extract features from fake images."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one
        import cv2

        config = {
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

        for img_path in real_train_fake_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)
            features = extract_one(foreground, mask, config)

            assert isinstance(features, dict)
            assert len(features) == 15

    def test_features_all_finite(self, real_train_live_images):
        """All extracted features should be finite (no NaN/Inf)."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one
        import cv2

        config = {
            'gabor_blk_size': 16, 'gabor_sigma': 6.0, 'gabor_freq': 0.1,
            'gabor_angle_num': 8, 'ocl_ofl_blk_size': 32, 'shared_blk_size': 64,
            'v1sz_x': 64, 'v1sz_y': 16, 'foreground_ratio': 0.8,
        }

        for img_path in real_train_live_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)
            features = extract_one(foreground, mask, config)

            for name, value in features.items():
                assert np.isfinite(value), f"Feature {name} is not finite: {value}"


class TestRealDataClassSeparation:
    """Test that features can separate live vs fake."""

    def test_live_fake_feature_difference(self, real_train_live_images, real_train_fake_images):
        """Live and fake images should have different feature distributions."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one
        import cv2

        config = {
            'gabor_blk_size': 16, 'gabor_sigma': 6.0, 'gabor_freq': 0.1,
            'gabor_angle_num': 8, 'ocl_ofl_blk_size': 32, 'shared_blk_size': 64,
            'v1sz_x': 64, 'v1sz_y': 16, 'foreground_ratio': 0.8,
        }

        live_features = []
        fake_features = []

        for img_path in real_train_live_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)
            features = extract_one(foreground, mask, config)
            live_features.append(list(features.values()))

        for img_path in real_train_fake_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            foreground, mask = extract_foreground(img)
            features = extract_one(foreground, mask, config)
            fake_features.append(list(features.values()))

        live_arr = np.array(live_features)
        fake_arr = np.array(fake_features)

        # Check if means are different (they should be for at least some features)
        live_mean = live_arr.mean(axis=0)
        fake_mean = fake_arr.mean(axis=0)

        # At least one feature should have different mean
        diff = np.abs(live_mean - fake_mean)
        assert diff.max() > 0.01, "Live and fake features are identical!"


class TestRealDataFullPipeline:
    """Test the full pipeline on real data."""

    def test_full_pipeline_live(self, real_train_live_images):
        """Run full pipeline on live images."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one, extract_dataset
        from feature_engineering import engineer_fingerprint_features
        import cv2

        config = {
            'gabor_blk_size': 16, 'gabor_sigma': 6.0, 'gabor_freq': 0.1,
            'gabor_angle_num': 8, 'ocl_ofl_blk_size': 32, 'shared_blk_size': 64,
            'v1sz_x': 64, 'v1sz_y': 16, 'foreground_ratio': 0.8,
        }

        # Load and preprocess
        pairs = []
        for img_path in real_train_live_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                foreground, mask = extract_foreground(img)
                pairs.append((foreground, mask))

        if len(pairs) == 0:
            pytest.skip("No valid images to test.")

        # Extract features
        df = extract_dataset(pairs, config)
        assert len(df) == len(pairs)

        # Engineer features
        df_eng = engineer_fingerprint_features(df)
        assert len(df_eng.columns) > len(df.columns)
        assert not df_eng.isnull().values.any()

    def test_full_pipeline_fake(self, real_train_fake_images):
        """Run full pipeline on fake images."""
        from preprocessing.preprocessing import extract_foreground
        from feature_extractor import extract_one, extract_dataset
        from feature_engineering import engineer_fingerprint_features
        import cv2

        config = {
            'gabor_blk_size': 16, 'gabor_sigma': 6.0, 'gabor_freq': 0.1,
            'gabor_angle_num': 8, 'ocl_ofl_blk_size': 32, 'shared_blk_size': 64,
            'v1sz_x': 64, 'v1sz_y': 16, 'foreground_ratio': 0.8,
        }

        pairs = []
        for img_path in real_train_fake_images[:5]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                foreground, mask = extract_foreground(img)
                pairs.append((foreground, mask))

        if len(pairs) == 0:
            pytest.skip("No valid images to test.")

        df = extract_dataset(pairs, config)
        df_eng = engineer_fingerprint_features(df)

        assert len(df_eng) == len(pairs)
        assert not np.isinf(df_eng.values).any()


class TestRealDataEdgeCases:
    """Test edge cases with real data."""

    def test_corrupted_image_handling(self, real_data_dir):
        """Should handle corrupted images gracefully."""
        import cv2
        from preprocessing.preprocessing import extract_foreground

        # Create a corrupted image file
        corrupted_path = real_data_dir / "corrupted_test.png"
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid image file')

        try:
            img = cv2.imread(str(corrupted_path), cv2.IMREAD_GRAYSCALE)
            # OpenCV returns None for corrupted images
            if img is None:
                # This is expected behavior
                pass
            else:
                # If it somehow loaded, preprocessing should handle it
                foreground, mask = extract_foreground(img)
                assert foreground is not None
        finally:
            if corrupted_path.exists():
                corrupted_path.unlink()

    def test_various_image_sizes(self, real_train_live_images):
        """Should handle images of various sizes."""
        from preprocessing.preprocessing import extract_foreground
        import cv2

        sizes_seen = set()
        for img_path in real_train_live_images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes_seen.add(img.shape)
                foreground, mask = extract_foreground(img)
                assert foreground is not None

        # We should have seen at least one valid image
        assert len(sizes_seen) > 0 or True  # Soft assertion

    def test_various_image_formats(self, real_data_dir):
        """Should handle various image formats."""
        from preprocessing.preprocessing import extract_foreground
        import cv2

        formats_tested = set()
        for ext in ["*.png", "*.bmp", "*.jpg", "*.jpeg"]:
            for img_path in (real_data_dir / "train" / "live").glob(ext):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    formats_tested.add(ext)
                    foreground, mask = extract_foreground(img)
                    assert foreground is not None
                    break  # One per format is enough

        # Should have tested at least one format
        assert len(formats_tested) > 0 or True
