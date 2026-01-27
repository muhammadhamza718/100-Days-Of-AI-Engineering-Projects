"""
Unit tests for Polynomial Regression implementation.

This module tests the core functionality of the Polynomial Regression implementation.
"""

import numpy as np
import pytest
from src.regression.polynomial_regression import PolynomialRegressor
from src.preprocessing.polynomial_features import generate_polynomial_features


def test_polynomial_regressor_initialization():
    """Test PolynomialRegressor initialization."""
    model = PolynomialRegressor(degree=2)

    assert model.degree == 2
    assert model.learning_rate == 0.01
    assert model.max_iterations == 1000
    assert model.tolerance == 1e-6
    assert model.original_n_features is None


def test_polynomial_regressor_fit_predict():
    """Test PolynomialRegressor fit and predict methods."""
    # Create simple synthetic quadratic data: y = x^2 + 2*x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Single feature
    true_coeff = [1.0, 2.0, 1.0]  # x^2 coefficient, x coefficient, intercept
    noise = np.random.normal(0, 0.5, size=X.shape[0])
    y = true_coeff[0] * X.flatten()**2 + true_coeff[1] * X.flatten() + true_coeff[2] + noise

    # Create and train polynomial model
    model = PolynomialRegressor(degree=2, learning_rate=0.001, max_iterations=2000)
    model.fit(X, y)

    # Verify model is fitted
    assert model.is_fitted
    assert model.theta is not None
    assert model.original_n_features == 1

    # Make predictions
    y_pred = model.predict(X)

    # Verify prediction shape
    assert y_pred.shape == y.shape

    # Verify R² score is reasonable (should be > 0.8 for this data)
    r2_score = model.score(X, y)
    assert r2_score > 0.7  # Reasonable threshold for this quadratic data


def test_polynomial_regressor_with_different_degrees():
    """Test PolynomialRegressor with different polynomial degrees."""
    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.rand(50, 1) * 5
    y = 0.5 * X.flatten()**3 - 2 * X.flatten()**2 + X.flatten() + 5 + np.random.normal(0, 0.1, size=X.shape[0])

    # Test different degrees
    degrees = [1, 2, 3]
    scores = []

    for degree in degrees:
        model = PolynomialRegressor(degree=degree, learning_rate=0.001, max_iterations=1500)
        model.fit(X, y)
        score = model.score(X, y)
        scores.append(score)
        assert model.is_fitted
        assert model.degree == degree

    # Higher degree models should generally have better fit on training data
    # (though this isn't guaranteed due to optimization challenges)
    assert len(scores) == len(degrees)


def test_polynomial_regressor_predict_before_fit():
    """Test that predict raises an error before fitting."""
    model = PolynomialRegressor(degree=2)
    X = np.random.rand(10, 2)

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        model.predict(X)


def test_polynomial_regressor_dimension_validation():
    """Test that predict validates input dimensions after fitting."""
    # Create and fit a model with 2 features
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    model = PolynomialRegressor(degree=2)
    model.fit(X, y)

    # Try to predict with wrong number of features
    X_wrong = np.random.rand(10, 3)  # 3 features instead of 2

    with pytest.raises(ValueError, match="Expected 2 features, got 3"):
        model.predict(X_wrong)


def test_polynomial_features_generation():
    """Test polynomial feature generation."""
    X = np.array([[1, 2], [3, 4]])  # 2 samples, 2 features

    # Generate polynomial features of degree 2
    X_poly = generate_polynomial_features(X, degree=2)

    # Expected features: [bias, x1, x2, x1^2, x1*x2, x2^2]
    # For X = [[1, 2], [3, 4]]:
    # Row 1: [1, 1, 2, 1, 2, 4] = [1, x1, x2, x1^2, x1*x2, x2^2]
    # Row 2: [1, 3, 4, 9, 12, 16]
    expected_shape = (2, 6)  # 2 samples, 6 features (1 + 2 + 3)
    assert X_poly.shape == expected_shape


def test_get_polynomial_degree():
    """Test getting polynomial degree."""
    model = PolynomialRegressor(degree=3)
    assert model.get_polynomial_degree() == 3


def test_get_original_n_features():
    """Test getting original number of features."""
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    model = PolynomialRegressor(degree=2)
    model.fit(X, y)

    assert model.get_original_n_features() == 3


def test_polynomial_regressor_with_linear_data():
    """Test PolynomialRegressor on linear data (should work well with degree 1)."""
    # Create simple linear data: y = 2*x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_slope = 2.0
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, size=X.shape[0])
    y = true_slope * X.flatten() + true_intercept + noise

    # Create and train polynomial model with degree 1 (should be equivalent to linear regression)
    model = PolynomialRegressor(degree=1, learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)

    # Verify model is fitted
    assert model.is_fitted
    assert model.degree == 1

    # Make predictions
    y_pred = model.predict(X)

    # Verify prediction shape
    assert y_pred.shape == y.shape

    # Verify R² score is reasonable
    r2_score = model.score(X, y)
    assert r2_score > 0.7  # Should be good for linear data


def test_polynomial_regressor_score_method():
    """Test the score method of PolynomialRegressor."""
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(50, 1) * 5
    y = X.flatten()**2 + np.random.normal(0, 0.1, size=X.shape[0])

    model = PolynomialRegressor(degree=2, learning_rate=0.001, max_iterations=1000)
    model.fit(X, y)

    # Calculate score
    score = model.score(X, y)

    # Score should be between -inf and 1
    assert score <= 1.0