"""
Unit tests for Ridge Regression implementation.

This module tests the core functionality of the Ridge Regression implementation.
"""

import numpy as np
import pytest
from src.regression.regularization.ridge_regression import RidgeRegressor
from src.data.housing_data import load_sample_housing_data
from src.preprocessing.feature_scaling import standardize_features


def test_ridge_regressor_initialization():
    """Test RidgeRegressor initialization with default parameters."""
    model = RidgeRegressor()

    assert model.lambda_reg == 0.01
    assert model.learning_rate == 0.01
    assert model.max_iterations == 1000
    assert model.tolerance == 1e-6
    assert model.theta is None
    assert model.cost_history is None
    assert not model.is_fitted


def test_ridge_regressor_custom_initialization():
    """Test RidgeRegressor initialization with custom parameters."""
    model = RidgeRegressor(lambda_reg=0.5, learning_rate=0.05, max_iterations=500, tolerance=1e-5)

    assert model.lambda_reg == 0.5
    assert model.learning_rate == 0.5
    assert model.max_iterations == 500
    assert model.tolerance == 1e-5


def test_ridge_regressor_fit_predict():
    """Test RidgeRegressor fit and predict methods."""
    # Create simple synthetic data: y = 2*x1 + 3*x2 + 1 + noise
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Two features
    true_weights = [2.0, 3.0]
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, size=X.shape[0])
    y = X[:, 0] * true_weights[0] + X[:, 1] * true_weights[1] + true_intercept + noise

    # Create and train model
    model = RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)

    # Verify model is fitted
    assert model.is_fitted
    assert model.theta is not None
    assert model.cost_history is not None
    assert len(model.cost_history) > 0

    # Make predictions
    y_pred = model.predict(X)

    # Verify prediction shape
    assert y_pred.shape == y.shape

    # Verify R² score is reasonable
    r2_score = model.score(X, y)
    assert r2_score > 0.7  # Reasonable threshold for this data


def test_ridge_regressor_with_high_regularization():
    """Test RidgeRegressor with high regularization (coefficients should be smaller)."""
    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(50, 3) * 10
    y = 2*X[:, 0] + 3*X[:, 1] + 1*X[:, 2] + 5 + np.random.normal(0, 0.5, size=X.shape[0])

    # Train with low regularization
    model_low_reg = RidgeRegressor(lambda_reg=0.01, learning_rate=0.01, max_iterations=1000)
    model_low_reg.fit(X, y)
    low_reg_params = model_low_reg.get_parameters()

    # Train with high regularization
    model_high_reg = RidgeRegressor(lambda_reg=10.0, learning_rate=0.01, max_iterations=1000)
    model_high_reg.fit(X, y)
    high_reg_params = model_high_reg.get_parameters()

    # High regularization should result in smaller (closer to 0) coefficients (excluding intercept)
    low_reg_weights_magnitude = np.mean(np.abs(low_reg_params[1:]))  # Exclude intercept
    high_reg_weights_magnitude = np.mean(np.abs(high_reg_params[1:]))  # Exclude intercept

    # High regularization should shrink coefficients
    assert high_reg_weights_magnitude <= low_reg_weights_magnitude


def test_ridge_regressor_with_multicollinear_data():
    """Test RidgeRegressor with multicollinear data (should handle better than linear regression)."""
    # Create synthetic data with multicollinearity
    np.random.seed(42)
    n_samples = 100

    # Create x1 randomly, then x2 as a linear combination of x1 + noise
    x1 = np.random.rand(n_samples) * 10
    x2 = 2 * x1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with x1
    X = np.column_stack([x1, x2])

    # Create target with both features
    y = 2*x1 + 1*x2 + 5 + np.random.normal(0, 0.5, size=n_samples)

    # Train Ridge model
    model = RidgeRegressor(lambda_reg=1.0, learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)

    # Verify model is fitted
    assert model.is_fitted
    assert model.theta is not None

    # Make predictions
    y_pred = model.predict(X)
    r2_score = model.score(X, y)

    # Should still achieve reasonable performance despite multicollinearity
    assert r2_score > 0.5


def test_ridge_regressor_predict_before_fit():
    """Test that predict raises an error before fitting."""
    model = RidgeRegressor()
    X = np.random.rand(10, 2)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.predict(X)


def test_ridge_regressor_score_before_fit():
    """Test that score raises an error before fitting."""
    model = RidgeRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.score(X, y)


def test_ridge_regressor_dimension_validation():
    """Test that fit validates input dimensions."""
    model = RidgeRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(8)  # Wrong size

    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        model.fit(X, y)


def test_get_regularization_strength():
    """Test getting regularization strength."""
    model = RidgeRegressor(lambda_reg=0.5)
    assert model.get_regularization_strength() == 0.5


def test_compute_ridge_cost():
    """Test the Ridge cost computation function."""
    from src.regression.regularization.ridge_regression import compute_ridge_cost

    # Simple test case
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5.5, 6.5])
    theta = np.array([1, 2])  # Intercept=1, slope=2
    lambda_reg = 0.1

    cost = compute_ridge_cost(X, y, theta, lambda_reg)

    # Should be MSE + L2 penalty
    # MSE component: predictions=[5, 7], errors=[0.5, -0.5], squared_errors=[0.25, 0.25], mse=0.25, half_mse=0.125
    # L2 penalty: (lambda_reg / (2*n)) * sum(theta[1:]^2) = (0.1 / (2*2)) * 4 = 0.1
    # Total: 0.125 + 0.1 = 0.225
    expected_cost = 0.225
    assert abs(cost - expected_cost) < 1e-5


def test_compute_ridge_gradients():
    """Test the Ridge gradient computation function."""
    from src.regression.regularization.ridge_regression import compute_ridge_gradients

    # Simple test case with perfect predictions
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5, 7])  # Perfect match for theta=[1, 2]
    theta = np.array([1, 2])  # Intercept=1, slope=2
    lambda_reg = 0.1

    gradients = compute_ridge_gradients(X, y, theta, lambda_reg)

    # With perfect predictions, MSE gradients would be [0, 0]
    # L2 regularization adds (lambda_reg / n) * theta to non-intercept terms
    # So gradient should be [0, (lambda_reg/n) * 2] = [0, (0.1/2) * 2] = [0, 0.1]
    expected_gradients = np.array([0.0, 0.1])
    np.testing.assert_array_almost_equal(gradients, expected_gradients, decimal=5)


def test_ridge_regressor_with_real_data():
    """Test RidgeRegressor with sample housing data."""
    # Load sample housing data
    X, y = load_sample_housing_data()

    # Scale features
    X_scaled, _, _ = standardize_features(X)

    # Create and train model
    model = RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
    model.fit(X_scaled, y)

    # Verify model is fitted
    assert model.is_fitted
    assert model.theta is not None
    assert len(model.cost_history) > 0

    # Make predictions on training data
    y_pred = model.predict(X_scaled)

    # Verify prediction shape
    assert y_pred.shape == y.shape

    # Verify R² score is reasonable
    r2_score = model.score(X_scaled, y)
    assert -1.0 <= r2_score <= 1.0  # R² should be in [-1, 1] range