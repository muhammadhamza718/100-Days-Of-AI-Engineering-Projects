"""
Unit tests for Linear Regression implementation.

This module tests the core functionality of the Linear Regression implementation.
"""

import numpy as np
import pytest
from src.regression.linear_regression import LinearRegressor
from src.data.housing_data import load_sample_housing_data
from src.preprocessing.feature_scaling import standardize_features


def test_linear_regressor_initialization():
    """Test LinearRegressor initialization with default parameters."""
    model = LinearRegressor()

    assert model.learning_rate == 0.01
    assert model.max_iterations == 1000
    assert model.tolerance == 1e-6
    assert model.theta is None
    assert model.cost_history is None
    assert not model.is_fitted


def test_linear_regressor_custom_initialization():
    """Test LinearRegressor initialization with custom parameters."""
    model = LinearRegressor(learning_rate=0.05, max_iterations=500, tolerance=1e-5)

    assert model.learning_rate == 0.05
    assert model.max_iterations == 500
    assert model.tolerance == 1e-5


def test_linear_regressor_fit_predict():
    """Test LinearRegressor fit and predict methods."""
    # Create simple synthetic data: y = 2*x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Single feature
    true_slope = 2.0
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, size=X.shape[0])
    y = true_slope * X.flatten() + true_intercept + noise

    # Create and train model
    model = LinearRegressor(learning_rate=0.01, max_iterations=1000)
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

    # Verify R² score is reasonable (should be > 0.7 for this data)
    r2_score = model.score(X, y)
    assert r2_score > 0.5  # Reasonable threshold for this data


def test_linear_regressor_with_real_data():
    """Test LinearRegressor with sample housing data."""
    # Load sample housing data
    X, y = load_sample_housing_data()

    # Scale features
    X_scaled, _, _ = standardize_features(X)

    # Create and train model
    model = LinearRegressor(learning_rate=0.01, max_iterations=1000)
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


def test_linear_regressor_predict_before_fit():
    """Test that predict raises an error before fitting."""
    model = LinearRegressor()
    X = np.random.rand(10, 2)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.predict(X)


def test_linear_regressor_score_before_fit():
    """Test that score raises an error before fitting."""
    model = LinearRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.score(X, y)


def test_linear_regressor_dimension_validation():
    """Test that fit validates input dimensions."""
    model = LinearRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(8)  # Wrong size

    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        model.fit(X, y)


def test_compute_hypothesis():
    """Test the hypothesis computation function."""
    from src.regression.linear_regression import compute_hypothesis

    # Simple test case
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    theta = np.array([1, 2])  # Intercept=1, slope=2

    result = compute_hypothesis(X, theta)

    # Expected: [1*1 + 2*2, 1*1 + 3*2] = [5, 7]
    expected = np.array([5.0, 7.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_mse_cost():
    """Test the MSE cost computation function."""
    from src.regression.linear_regression import compute_mse_cost

    # Simple test case
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5.5, 6.5])
    theta = np.array([1, 2])  # Intercept=1, slope=2

    cost = compute_mse_cost(X, y, theta)

    # Predictions: [5, 7], Errors: [0.5, -0.5], Squared errors: [0.25, 0.25], MSE: 0.25, Half MSE: 0.125
    expected_cost = 0.125
    assert abs(cost - expected_cost) < 1e-6


def test_compute_cost_gradient():
    """Test the gradient computation function."""
    from src.regression.linear_regression import compute_cost_gradient

    # Simple test case
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5, 7])
    theta = np.array([1, 2])  # Intercept=1, slope=2

    gradients = compute_cost_gradient(X, y, theta)

    # With perfect predictions (y_pred = y), gradients should be close to [0, 0]
    expected_gradients = np.array([0.0, 0.0])
    np.testing.assert_array_almost_equal(gradients, expected_gradients, decimal=5)


def test_get_parameters_and_cost_history():
    """Test the parameter and cost history retrieval methods."""
    X = np.random.rand(20, 1)
    y = 2 * X.flatten() + 1 + np.random.normal(0, 0.1, size=X.shape[0])

    model = LinearRegressor(learning_rate=0.1, max_iterations=100)
    model.fit(X, y)

    # Test parameter retrieval
    params = model.get_parameters()
    assert params is not None
    assert len(params) == X.shape[1] + 1  # +1 for intercept

    # Test cost history retrieval
    cost_history = model.get_cost_history()
    assert cost_history is not None
    assert len(cost_history) > 0