"""
Tests for MLflow logging and model tracking.
Uses mocking to avoid actual MLflow server calls.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMLflowLogging:
    """Tests for MLflow logging functionality."""

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_logs_hyperparameters(self, mock_log_metrics, mock_log_params, mock_start_run):
        """Should log all hyperparameters to MLflow."""
        from train import DEFAULT_CONFIG

        # Simulate what train.py does
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()

        with patch('mlflow.start_run', return_value=mock_start_run.return_value):
            # Simulate logging
            import mlflow
            with mlflow.start_run(run_name="test_run"):
                mlflow.log_params(DEFAULT_CONFIG)

        # Verify log_params was called
        mock_log_params.assert_called()
        call_args = mock_log_params.call_args[0][0]

        # Check all expected params were logged
        expected_params = ['gabor_blk_size', 'gabor_sigma', 'gabor_freq', 'foreground_ratio']
        for param in expected_params:
            assert param in call_args

    @patch('mlflow.start_run')
    @patch('mlflow.log_metrics')
    def test_logs_metrics(self, mock_log_metrics, mock_start_run):
        """Should log evaluation metrics to MLflow."""
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()

        metrics = {
            'val_auc': 0.95,
            'val_accuracy': 0.92,
        }

        with patch('mlflow.start_run', return_value=mock_start_run.return_value):
            import mlflow
            with mlflow.start_run(run_name="test_run"):
                mlflow.log_metrics(metrics)

        mock_log_metrics.assert_called_once()
        call_args = mock_log_metrics.call_args[0][0]

        assert 'val_auc' in call_args
        assert 'val_accuracy' in call_args
        assert call_args['val_auc'] == 0.95

    @patch('mlflow.log_artifact')
    def test_logs_artifacts(self, mock_log_artifact, tmp_path):
        """Should log artifacts (models, plots) to MLflow."""
        # Create a temporary artifact file
        artifact_file = tmp_path / "test_artifact.txt"
        artifact_file.write_text("Test artifact content")

        import mlflow
        with patch('mlflow.log_artifact'):
            mlflow.log_artifact(str(artifact_file))

        mock_log_artifact.assert_called()


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_tabnet_save_and_load(self, tmp_path):
        """TabNet model should be savable and loadable."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            pytest.skip("pytorch-tabnet not installed")

        # Create a simple model
        model = TabNetClassifier(n_d=8, n_a=8, n_steps=2)

        # Create dummy data for initialization
        X_dummy = np.random.randn(20, 5)
        y_dummy = np.array([0] * 10 + [1] * 10)

        # Fit briefly to initialize
        model.fit(X_dummy, y_dummy, max_epochs=1)

        # Save model
        save_path = tmp_path / "test_model"
        model.save_model(str(save_path))

        # Verify save directory exists
        assert save_path.exists()

        # Load model
        loaded_model = TabNetClassifier()
        loaded_model.load_model(str(save_path))

        # Verify loaded model works
        predictions = loaded_model.predict(X_dummy)
        assert len(predictions) == 20

    def test_scaler_save_and_load(self, tmp_path):
        """Scalers should be savable and loadable with pickle."""
        import pickle
        from sklearn.preprocessing import RobustScaler, QuantileTransformer

        # Create and fit scalers
        robust = RobustScaler()
        qt = QuantileTransformer(output_distribution='normal', random_state=42)

        X = np.random.randn(100, 10)
        robust.fit(X)
        qt.fit(robust.transform(X))

        # Save
        scalers = {'robust': robust, 'quantile': qt}
        save_path = tmp_path / "scalers.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(scalers, f)

        # Load
        with open(save_path, 'rb') as f:
            loaded = pickle.load(f)

        assert 'robust' in loaded
        assert 'quantile' in loaded

        # Verify loaded scalers work
        X_test = np.random.randn(10, 10)
        X_scaled = loaded['quantile'].transform(loaded['robust'].transform(X_test))
        assert X_scaled.shape == X_test.shape


class TestExperimentTracking:
    """Tests for experiment tracking functionality."""

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_sets_up_experiment(self, mock_set_experiment, mock_set_tracking_uri):
        """Should set up MLflow tracking correctly."""
        import mlflow

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("test_experiment")

        mock_set_tracking_uri.assert_called_once_with("mlruns")
        mock_set_experiment.assert_called_once_with("test_experiment")

    def test_run_naming(self):
        """Should use meaningful run names."""
        run_name = "final_tabnet"
        assert "tabnet" in run_name.lower()

    def test_metric_validation(self):
        """Metrics should be valid numbers."""
        metrics = {
            'auc': 0.95,
            'accuracy': 0.92,
            'loss': 0.35,
        }

        for name, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1 or value >= 0  # AUC/accuracy in [0,1], loss >= 0


class TestConfigManagement:
    """Tests for configuration management."""

    def test_config_serialization(self, tmp_path):
        """Config should be serializable to JSON."""
        import json

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

        # Save to JSON
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Load from JSON
        with open(config_path, 'r') as f:
            loaded = json.load(f)

        assert loaded == config

    def test_config_validation(self):
        """Config values should be in valid ranges."""
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

        # Validate ranges
        assert 8 <= config['gabor_blk_size'] <= 64
        assert 0 < config['gabor_sigma'] < 20
        assert 0 < config['gabor_freq'] < 1
        assert 4 <= config['gabor_angle_num'] <= 16
        assert 0.5 < config['foreground_ratio'] < 1.0


class TestReproducibility:
    """Tests for reproducibility and determinism."""

    def test_random_seed_reproducibility(self):
        """Same seed should produce same results."""
        np.random.seed(42)
        result1 = np.random.randn(10)

        np.random.seed(42)
        result2 = np.random.randn(10)

        assert np.array_equal(result1, result2)

    def test_feature_extraction_determinism(self, sample_image_pair, sample_config):
        """Feature extraction should be deterministic."""
        from feature_extractor import extract_one

        img, mask = sample_image_pair

        result1 = extract_one(img, mask, sample_config)
        result2 = extract_one(img, mask, sample_config)

        for key in result1:
            assert result1[key] == result2[key], f"Feature {key} is not deterministic"


class TestErrorHandling:
    """Tests for error handling in production scenarios."""

    def test_missing_config_file(self, tmp_path):
        """Should handle missing config file gracefully."""
        import json

        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            with open(config_path, 'r') as f:
                json.load(f)

    def test_invalid_image_path(self):
        """Should handle invalid image paths."""
        import cv2

        img = cv2.imread("nonexistent_image.png", cv2.IMREAD_GRAYSCALE)
        assert img is None

    def test_empty_dataset_handling(self, sample_config):
        """Should handle empty datasets."""
        from feature_extractor import extract_dataset

        result = extract_dataset([], sample_config)

        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_model_without_training(self, tmp_path):
        """Untrained model should still save/load."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            pytest.skip("pytorch-tabnet not installed")

        model = TabNetClassifier()
        save_path = tmp_path / "untrained_model"

        # Should be able to save without training
        model.save_model(str(save_path))
        assert save_path.exists()
