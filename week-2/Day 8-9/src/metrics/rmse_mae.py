"""
Additional metrics: RMSE, MAE, and related implementations.

This module provides additional regression metrics beyond MSE and R-squared,
including Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
"""

import numpy as np
from typing import Tuple, Optional
from src.metrics.mse import calculate_mse, calculate_mae
from src.metrics.r_squared import calculate_r_squared


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    RMSE is calculated as: RMSE = sqrt(MSE) = sqrt((1/n) * Σ(y_true - y_pred)²)
    RMSE is in the same units as the target variable and gives more weight to larger errors.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Root Mean Squared Error
    """
    # Calculate MSE first
    mse = calculate_mse(y_true, y_pred)

    # Calculate RMSE as square root of MSE
    rmse = np.sqrt(mse)

    return rmse


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    MAE is calculated as: MAE = (1/n) * Σ|y_true - y_pred|
    MAE is in the same units as the target variable and is robust to outliers.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Mean Absolute Error
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Calculate mean of absolute errors
    mae = np.mean(absolute_errors)

    return mae


def calculate_metrics_comprehensive(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> dict:
    """
    Calculate a comprehensive set of regression metrics.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        n_features (Optional[int]): Number of features for adjusted R-squared calculation

    Returns:
        dict: Dictionary containing comprehensive regression metrics
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate individual metrics
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r_squared = calculate_r_squared(y_true, y_pred)

    # Calculate additional metrics
    # Mean Absolute Percentage Error (avoiding division by zero)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')

    # Mean Percentage Error (can be negative)
    if np.any(non_zero_mask):
        mpe = np.mean(((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mpe = float('inf')

    # Calculate standard deviation of errors
    errors = y_true - y_pred
    error_std = np.std(errors)

    # Calculate median absolute error (robust to outliers)
    median_abs_error = np.median(np.abs(errors))

    # Calculate max error
    max_error = np.max(np.abs(errors))

    # Calculate variance explained (similar to R²)
    total_variance = np.var(y_true)
    error_variance = np.var(errors)
    if total_variance == 0:
        variance_explained = 1.0 if error_variance == 0 else 0.0
    else:
        variance_explained = 1 - (error_variance / total_variance)

    # Calculate adjusted R² if n_features provided
    if n_features is not None and len(y_true) > n_features + 1:
        n = len(y_true)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)
    else:
        adj_r_squared = None

    return {
        # Basic metrics
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'adjusted_r_squared': adj_r_squared,

        # Additional metrics
        'mape': mape,
        'mpe': mpe,
        'median_absolute_error': median_abs_error,
        'max_error': max_error,
        'error_std': error_std,
        'variance_explained': variance_explained,

        # Data info
        'n_observations': len(y_true),
        'n_features': n_features,
        'mean_true': np.mean(y_true),
        'std_true': np.std(y_true),
        'mean_pred': np.mean(y_pred),
        'std_pred': np.std(y_pred)
    }


def calculate_error_quantiles(y_true: np.ndarray, y_pred: np.ndarray, quantiles: list = [0.25, 0.5, 0.75]) -> dict:
    """
    Calculate quantiles of the error distribution.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        quantiles (list): List of quantiles to calculate (default: [0.25, 0.5, 0.75])

    Returns:
        dict: Dictionary containing error quantiles
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate errors
    errors = y_true - y_pred

    # Calculate specified quantiles
    quantile_results = {}
    for q in quantiles:
        quantile_results[f'error_q{int(q*100)}'] = np.quantile(errors, q)

    # Add min and max errors
    quantile_results['error_min'] = np.min(errors)
    quantile_results['error_max'] = np.max(errors)

    return quantile_results


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy - the percentage of times the model correctly
    predicts the direction of change from the previous value.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Directional accuracy (0.0 to 1.0)
    """
    if len(y_true) < 2:
        return 0.0

    # Calculate directional changes
    true_directions = np.diff(y_true)  # true_t+1 - true_t
    pred_directions = np.diff(y_pred)  # pred_t+1 - pred_t

    # Count how many times directions match (excluding zero changes)
    same_direction = (true_directions * pred_directions) > 0
    zero_changes = (true_directions == 0) | (pred_directions == 0)

    # Calculate directional accuracy
    if len(same_direction) == 0:
        return 0.0

    # Count correct predictions excluding zero changes
    correct_directions = np.sum(same_direction & ~zero_changes)
    total_comparable = np.sum(~zero_changes)

    if total_comparable == 0:
        return 0.0

    directional_accuracy = correct_directions / total_comparable

    return directional_accuracy


def calculate_theil_inequality_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Theil's Inequality Coefficient, which measures forecast accuracy.
    Value ranges from 0 (perfect forecast) to 1 (worst possible forecast).

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Theil's Inequality Coefficient
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    if len(y_true) == 0:
        return float('nan')

    # Calculate the three components of the Theil coefficient
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom_term1 = np.sqrt(np.mean(y_true ** 2))
    denom_term2 = np.sqrt(np.mean(y_pred ** 2))

    # Calculate the Theil coefficient
    denominator = denom_term1 + denom_term2
    if denominator == 0:
        return 0.0  # Perfect prediction

    theil_coefficient = numerator / denominator

    return theil_coefficient


def calculate_metrics_by_percentile(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> list:
    """
    Calculate metrics for different percentiles/ranges of the target variable.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        n_bins (int): Number of bins to divide the data into

    Returns:
        list: List of dictionaries containing metrics for each percentile bin
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Sort indices based on true values
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Divide into bins
    n_samples = len(y_true)
    bin_size = n_samples // n_bins

    results = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

        y_true_bin = y_true_sorted[start_idx:end_idx]
        y_pred_bin = y_pred_sorted[start_idx:end_idx]

        if len(y_true_bin) > 0:
            bin_result = {
                'bin_index': i,
                'start_value': y_true_sorted[start_idx],
                'end_value': y_true_sorted[end_idx - 1] if end_idx > start_idx else y_true_sorted[start_idx],
                'n_samples': len(y_true_bin),
                'mse': calculate_mse(y_true_bin, y_pred_bin),
                'rmse': calculate_rmse(y_true_bin, y_pred_bin),
                'mae': calculate_mae(y_true_bin, y_pred_bin),
                'r_squared': calculate_r_squared(y_true_bin, y_pred_bin),
                'mean_true': np.mean(y_true_bin),
                'mean_pred': np.mean(y_pred_bin)
            }
            results.append(bin_result)

    return results