"""
Unit tests for metrics implementation.

This module tests the core functionality of the evaluation metrics.
"""

import numpy as np
import pytest
from src.metrics.mse import calculate_mse, calculate_rmse, calculate_mae
from src.metrics.r_squared import calculate_r_squared, calculate_adjusted_r_squared
from src.metrics.rmse_mae import calculate_metrics_comprehensive


def test_calculate_mse():
    """Test Mean Squared Error calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mse = calculate_mse(y_true, y_pred)

    # Expected: [(3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2] / 4
    # = [0.25 + 0.25 + 0 + 1] / 4 = 1.5 / 4 = 0.375
    expected_mse = 0.375
    assert abs(mse - expected_mse) < 1e-10

    # Test with identical arrays (MSE should be 0)
    mse_zero = calculate_mse(y_true, y_true)
    assert mse_zero == 0.0


def test_calculate_mse_with_invalid_shapes():
    """Test MSE calculation with mismatched shapes."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])

    with pytest.raises(ValueError, match="y_true and y_pred must have the same shape"):
        calculate_mse(y_true, y_pred)


def test_calculate_rmse():
    """Test Root Mean Squared Error calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    rmse = calculate_rmse(y_true, y_pred)
    mse = calculate_mse(y_true, y_pred)

    # RMSE should be the square root of MSE
    expected_rmse = np.sqrt(mse)
    assert abs(rmse - expected_rmse) < 1e-10

    # Test with identical arrays (RMSE should be 0)
    rmse_zero = calculate_rmse(y_true, y_true)
    assert rmse_zero == 0.0


def test_calculate_mae():
    """Test Mean Absolute Error calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mae = calculate_mae(y_true, y_pred)

    # Expected: [|3-2.5| + |-0.5-0| + |2-2| + |7-8|] / 4
    # = [0.5 + 0.5 + 0 + 1] / 4 = 2 / 4 = 0.5
    expected_mae = 0.5
    assert abs(mae - expected_mae) < 1e-10

    # Test with identical arrays (MAE should be 0)
    mae_zero = calculate_mae(y_true, y_true)
    assert mae_zero == 0.0


def test_calculate_r_squared():
    """Test R-squared calculation."""
    # Perfect prediction should give R² = 1
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    r2_perfect = calculate_r_squared(y_true, y_pred)
    assert abs(r2_perfect - 1.0) < 1e-10

    # Test with random data
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    r2 = calculate_r_squared(y_true, y_pred)

    # Verify R² is in valid range
    assert -1.0 <= r2 <= 1.0


def test_calculate_r_squared_edge_cases():
    """Test R-squared calculation with edge cases."""
    # All true values are the same
    y_true = np.array([5, 5, 5, 5])
    y_pred = np.array([4, 6, 5, 5])

    # If all true values are the same, R² should be 0 if predictions vary
    r2 = calculate_r_squared(y_true, y_pred)
    # Since SS_tot is 0, the formula becomes 1 - (SS_res / 0), which is undefined
    # Our implementation should handle this case appropriately
    assert isinstance(r2, float)


def test_calculate_adjusted_r_squared():
    """Test Adjusted R-squared calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    n_features = 2

    adj_r2 = calculate_adjusted_r_squared(y_true, y_pred, n_features)

    # Verify adjusted R² is a float
    assert isinstance(adj_r2, float)

    # Adjusted R² should be <= regular R²
    r2 = calculate_r_squared(y_true, y_pred)
    if adj_r2 is not None and not np.isinf(adj_r2):
        assert adj_r2 <= r2 or (np.isinf(adj_r2) and adj_r2 < 0)


def test_calculate_adjusted_r_squared_invalid_n_features():
    """Test Adjusted R-squared with invalid number of features."""
    y_true = np.array([1, 2, 3])  # 3 observations
    y_pred = np.array([1, 2, 3])
    n_features = 3  # Same as number of observations (invalid)

    adj_r2 = calculate_adjusted_r_squared(y_true, y_pred, n_features)

    # Should return negative infinity when n <= k + 1
    assert np.isinf(adj_r2) and adj_r2 < 0


def test_calculate_metrics_comprehensive():
    """Test comprehensive metrics calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    n_features = 1

    metrics = calculate_metrics_comprehensive(y_true, y_pred, n_features)

    # Verify all expected keys are present
    expected_keys = [
        'mse', 'rmse', 'mae', 'r_squared', 'adjusted_r_squared',
        'mape', 'mpe', 'median_absolute_error', 'max_error', 'error_std',
        'variance_explained', 'n_observations', 'n_features', 'mean_true',
        'std_true', 'mean_pred', 'std_pred'
    ]

    for key in expected_keys:
        assert key in metrics

    # Verify some specific values
    assert metrics['n_observations'] == len(y_true)
    assert metrics['n_features'] == n_features
    assert abs(metrics['mse'] - calculate_mse(y_true, y_pred)) < 1e-10


def test_calculate_metrics_comprehensive_without_n_features():
    """Test comprehensive metrics calculation without n_features."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    metrics = calculate_metrics_comprehensive(y_true, y_pred)

    # Verify adjusted R² is None when n_features is not provided
    assert metrics['adjusted_r_squared'] is None


def test_metrics_with_constant_predictions():
    """Test metrics with constant predictions."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([3, 3, 3, 3, 3])  # Constant predictions

    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r_squared(y_true, y_pred)

    # Verify calculations are valid
    assert mse > 0  # Should have some error
    assert rmse == np.sqrt(mse)
    assert mae > 0  # Should have some error
    assert isinstance(r2, float)


def test_metrics_with_large_values():
    """Test metrics with large values."""
    y_true = np.array([1000000, 2000000, 3000000])
    y_pred = np.array([1000001, 2000002, 2999999])

    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    # Verify calculations are reasonable
    assert mse >= 0
    assert rmse >= 0
    assert mae >= 0
    assert rmse == np.sqrt(mse)


def test_metrics_with_negative_values():
    """Test metrics with negative values."""
    y_true = np.array([-1, -2, -3, -4])
    y_pred = np.array([-1.1, -1.9, -3.1, -3.9])

    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    # Verify calculations are valid (metrics should be non-negative)
    assert mse >= 0
    assert rmse >= 0
    assert mae >= 0
    assert rmse == np.sqrt(mse)