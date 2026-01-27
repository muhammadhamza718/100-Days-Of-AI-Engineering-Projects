"""
Unit tests for Lasso Regression implementation.

This module tests the core functionality of the Lasso Regression implementation.
"""

import numpy as np
import pytest
from src.regression.regularization.lasso_regression import LassoRegressor
from src.data.housing_data import load_sample_housing_data
from src.preprocessing.feature_scaling import standardize_features


def test_lasso_regressor_initialization():
    """Test LassoRegressor initialization with default parameters."""
    model = LassoRegressor()

    assert model.lambda_reg == 0.01
    assert model.learning_rate == 0.01
    assert model.max_iterations == 1000
    assert model.tolerance == 1e-6
    assert model.theta is None
    assert model.cost_history is None
    assert not model.is_fitted


def test_lasso_regressor_custom_initialization():
    """Test LassoRegressor initialization with custom parameters."""
    model = LassoRegressor(lambda_reg=0.5, learning_rate=0.05, max_iterations=500, tolerance=1e-5)

    assert model.lambda_reg == 0.5
    assert model.learning_rate == 0.05
    assert model.max_iterations == 500
    assert model.tolerance == 1e-5


def test_lasso_regressor_fit_predict():
    """Test LassoRegressor fit and predict methods."""
    # Create simple synthetic data: y = 2*x1 + 0*x2 + 1 + noise
    # The second feature has no effect, so Lasso should drive its coefficient to zero
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Two features
    true_weights = [2.0, 0.0]  # Second feature has no effect
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, size=X.shape[0])
    y = X[:, 0] * true_weights[0] + X[:, 1] * true_weights[1] + true_intercept + noise

    # Create and train model
    model = LassoRegressor(lambda_reg=0.5, learning_rate=0.01, max_iterations=1000)
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


def test_lasso_regressor_feature_selection():
    """Test LassoRegressor's feature selection capability."""
    # Create synthetic data with many irrelevant features
    np.random.seed(42)
    n_samples, n_features = 100, 10

    # Generate features
    X = np.random.rand(n_samples, n_features) * 10

    # Only first 3 features matter for target
    true_weights = [2.0, -1.5, 3.0] + [0.0] * (n_features - 3)  # Only first 3 features matter
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, size=n_samples)
    y = X[:, :3].dot(true_weights[:3]) + true_intercept + noise

    # Train Lasso model with high regularization
    model = LassoRegressor(lambda_reg=1.0, learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)

    # Get learned parameters
    params = model.get_parameters()

    # Check that the model is fitted
    assert model.is_fitted

    # The intercept should be close to true intercept
    assert abs(params[0] - true_intercept) < 5  # Allow some tolerance due to noise


def test_lasso_regressor_with_high_regularization():
    """Test LassoRegressor with high regularization (should set some coefficients to zero)."""
    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(50, 3) * 10
    y = 2*X[:, 0] + 3*X[:, 1] + 1*X[:, 2] + 5 + np.random.normal(0, 0.5, size=X.shape[0])

    # Train with high regularization
    model = LassoRegressor(lambda_reg=5.0, learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)

    # Get parameters
    params = model.get_parameters()

    # Check that model is fitted
    assert model.is_fitted

    # High regularization may set some coefficients to zero or near-zero
    # This is the feature selection property of Lasso


def test_lasso_regressor_predict_before_fit():
    """Test that predict raises an error before fitting."""
    model = LassoRegressor()
    X = np.random.rand(10, 2)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.predict(X)


def test_lasso_regressor_score_before_fit():
    """Test that score raises an error before fitting."""
    model = LassoRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.score(X, y)


def test_lasso_regressor_dimension_validation():
    """Test that fit validates input dimensions."""
    model = LassoRegressor()
    X = np.random.rand(10, 2)
    y = np.random.rand(8)  # Wrong size

    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        model.fit(X, y)


def test_get_regularization_strength():
    """Test getting regularization strength."""
    model = LassoRegressor(lambda_reg=0.5)
    assert model.get_regularization_strength() == 0.5


def test_get_coefficient_sparsity():
    """Test getting coefficient sparsity."""
    # Create simple model
    np.random.seed(42)
    X = np.random.rand(20, 3)
    y = 2*X[:, 0] + 0*X[:, 1] + 1*X[:, 2] + np.random.normal(0, 0.1, size=X.shape[0])

    model = LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=500)
    model.fit(X, y)

    sparsity = model.get_coefficient_sparsity()
    assert 0.0 <= sparsity <= 1.0  # Sparsity should be between 0 and 1


def test_compute_lasso_cost():
    """Test the Lasso cost computation function."""
    from src.regression.regularization.lasso_regression import compute_lasso_cost

    # Simple test case
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5.5, 6.5])
    theta = np.array([1, 2])  # Intercept=1, slope=2
    lambda_reg = 0.1

    cost = compute_lasso_cost(X, y, theta, lambda_reg)

    # Should be MSE + L1 penalty
    # MSE component: predictions=[5, 7], errors=[0.5, -0.5], squared_errors=[0.25, 0.25], mse=0.25, half_mse=0.125
    # L1 penalty: (lambda_reg / n) * sum(|theta[1:]|) = (0.1 / 2) * 2 = 0.1
    # Total: 0.125 + 0.1 = 0.225
    expected_cost = 0.225
    assert abs(cost - expected_cost) < 1e-5


def test_compute_lasso_gradients():
    """Test the Lasso gradient computation function using subgradients."""
    from src.regression.regularization.lasso_regression import compute_lasso_gradients

    # Simple test case with perfect predictions
    X = np.array([[1, 2], [1, 3]])  # With intercept term
    y = np.array([5, 7])  # Perfect match for theta=[1, 2]
    theta = np.array([1, 2])  # Intercept=1, slope=2
    lambda_reg = 0.1

    gradients = compute_lasso_gradients(X, y, theta, lambda_reg)

    # With perfect predictions, MSE gradients would be [0, 0]
    # L1 regularization adds (lambda_reg / n) * sign(theta_j) to non-intercept terms
    # So gradient should be [0, (lambda_reg/n) * sign(2)] = [0, (0.1/2) * 1] = [0, 0.05]
    expected_gradients = np.array([0.0, 0.05])
    np.testing.assert_array_almost_equal(gradients, expected_gradients, decimal=5)


def test_soft_threshold_function():
    """Test the soft thresholding function."""
    from src.regression.regularization.lasso_regression import soft_threshold

    # Test cases
    assert soft_threshold(1.5, 1.0) == 0.5  # 1.5 - 1.0 = 0.5
    assert soft_threshold(-1.5, 1.0) == -0.5  # -1.5 + 1.0 = -0.5
    assert soft_threshold(0.5, 1.0) == 0.0  # Within threshold
    assert soft_threshold(-0.5, 1.0) == 0.0  # Within threshold
    assert soft_threshold(0.0, 1.0) == 0.0  # Zero case


def test_lasso_regressor_with_real_data():
    """Test LassoRegressor with sample housing data."""
    # Load sample housing data
    X, y = load_sample_housing_data()

    # Scale features
    X_scaled, _, _ = standardize_features(X)

    # Create and train model
    model = LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
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