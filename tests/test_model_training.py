"""Tests for model training functions."""

import pytest
import pandas as pd
import numpy as np

from src.data_processing import train_model
import xgboost as xgb


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        # Create random features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Create target with some relationship to features
        y = pd.Series(
            X["feature_0"] * 2 + X["feature_1"] * 3 + np.random.randn(n_samples) * 0.5
        )

        return X, y

    @pytest.mark.unit
    def test_returns_correct_types(self, sample_training_data):
        """Test that function returns correct types."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Check model type
        assert isinstance(model, xgb.XGBRegressor)

        # Check data splits
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    @pytest.mark.unit
    def test_train_test_split_sizes(self, sample_training_data):
        """Test that train/test split has correct proportions."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Test size should be 30% (TEST_SIZE = 0.3)
        total_samples = len(X)
        test_samples = len(X_test)

        # Allow small margin for rounding
        expected_test_size = int(total_samples * 0.3)
        assert abs(test_samples - expected_test_size) <= 1

        # Train + test should equal total
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples

    @pytest.mark.unit
    def test_model_is_fitted(self, sample_training_data):
        """Test that returned model is fitted."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Fitted model should have these attributes
        assert hasattr(model, "feature_importances_")
        assert hasattr(model, "n_features_in_")

        # Should be able to make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    @pytest.mark.unit
    def test_model_predictions_reasonable(self, sample_training_data):
        """Test that model predictions are reasonable."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        predictions = model.predict(X_test)

        # Predictions should be numeric
        assert np.isfinite(predictions).all()

        # Predictions should be in reasonable range relative to target
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min

        # Predictions should be within expanded range (model might extrapolate slightly)
        assert predictions.min() >= y_min - y_range
        assert predictions.max() <= y_max + y_range

    @pytest.mark.unit
    def test_feature_importances_exist(self, sample_training_data):
        """Test that model has feature importances."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        importances = model.feature_importances_

        # Should have importance for each feature
        assert len(importances) == X.shape[1]

        # Importances should be non-negative
        assert (importances >= 0).all()

        # Importances should sum to approximately 1
        assert abs(importances.sum() - 1.0) < 0.01

    @pytest.mark.unit
    def test_reproducibility(self, sample_training_data):
        """Test that model training is reproducible with same random state."""
        X, y = sample_training_data

        # Train model twice
        model1, X_train1, X_test1, y_train1, y_test1 = train_model(X, y)
        model2, X_train2, X_test2, y_train2, y_test2 = train_model(X, y)

        # Splits should be identical (same random state)
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

        # Predictions should be identical
        pred1 = model1.predict(X_test1)
        pred2 = model2.predict(X_test2)
        np.testing.assert_array_almost_equal(pred1, pred2)

    @pytest.mark.unit
    def test_handles_small_dataset(self):
        """Test that model can handle small datasets."""
        # Create minimal dataset (10 samples)
        X = pd.DataFrame(
            {
                "feature_0": np.random.randn(10),
                "feature_1": np.random.randn(10),
                "feature_2": np.random.randn(10),
            }
        )
        y = pd.Series(np.random.randn(10))

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Should still work
        assert isinstance(model, xgb.XGBRegressor)
        assert len(X_train) > 0
        assert len(X_test) > 0

    @pytest.mark.unit
    def test_model_objective(self, sample_training_data):
        """Test that model uses correct objective function."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # Should be using squared error for regression
        # XGBoost stores this in the model's parameters
        assert model.objective == "reg:squarederror"

    @pytest.mark.unit
    def test_no_data_leakage(self, sample_training_data):
        """Test that there's no data leakage between train and test sets."""
        X, y = sample_training_data

        model, X_train, X_test, y_train, y_test = train_model(X, y)

        # No overlap between train and test indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0

    @pytest.mark.unit
    def test_handles_different_feature_counts(self):
        """Test model training with different numbers of features."""
        feature_counts = [5, 10, 20, 50]

        for n_features in feature_counts:
            X = pd.DataFrame(
                np.random.randn(100, n_features),
                columns=[f"feature_{i}" for i in range(n_features)],
            )
            y = pd.Series(np.random.randn(100))

            model, X_train, X_test, y_train, y_test = train_model(X, y)

            assert model.n_features_in_ == n_features
            assert len(model.feature_importances_) == n_features
