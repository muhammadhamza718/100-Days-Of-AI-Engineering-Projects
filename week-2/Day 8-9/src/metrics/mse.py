"""
Mean Squared Error (MSE) metric implementation.

This module provides functions to calculate the Mean Squared Error,
which is a common metric for evaluating regression model performance.
"""

import numpy as np
from typing import Union


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    MSE is calculated as: MSE = (1/n) * Σ(y_true - y_pred)²

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Mean Squared Error
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Calculate mean of squared errors
    mse = np.mean(squared_errors)

    return mse


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    RMSE is calculated as: RMSE = sqrt(MSE) = sqrt((1/n) * Σ(y_true - y_pred)²)

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


def calculate_mse_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Decompose the MSE into bias and variance components.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        dict: Dictionary containing MSE decomposition:
              - 'total_mse': Total MSE
              - 'bias_squared': Squared bias component
              - 'variance': Variance component
              - 'irreducible_error': Irreducible error component
    """
    # Calculate total MSE
    total_mse = calculate_mse(y_true, y_pred)

    # Calculate bias squared: (E[y_pred] - E[y_true])^2
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    bias_squared = (mean_pred - mean_true) ** 2

    # Calculate variance: E[(y_pred - E[y_pred])^2]
    variance = np.mean((y_pred - mean_pred) ** 2)

    # Calculate irreducible error: MSE - bias^2 - variance
    irreducible_error = total_mse - bias_squared - variance

    return {
        'total_mse': total_mse,
        'bias_squared': bias_squared,
        'variance': variance,
        'irreducible_error': irreducible_error
    }


def calculate_mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Logarithmic Error (MSLE).

    MSLE is calculated as: MSLE = (1/n) * Σ(log(y_true + 1) - log(y_pred + 1))²

    Args:
        y_true (np.ndarray): True target values (must be non-negative)
        y_pred (np.ndarray): Predicted target values (must be non-negative)

    Returns:
        float: Mean Squared Logarithmic Error
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Ensure non-negative values to avoid log(0) or log(negative)
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("y_true and y_pred must contain non-negative values for MSLE")

    # Calculate logarithms
    log_y_true = np.log(y_true + 1)
    log_y_pred = np.log(y_pred + 1)

    # Calculate squared log errors
    squared_log_errors = (log_y_true - log_y_pred) ** 2

    # Calculate mean of squared log errors
    msle = np.mean(squared_log_errors)

    return msle


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    MAPE is calculated as: MAPE = (1/n) * Σ|(y_true - y_pred) / y_true| * 100

    Args:
        y_true (np.ndarray): True target values (must be non-zero)
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Mean Absolute Percentage Error
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Ensure no zero values in y_true to avoid division by zero
    if np.any(y_true == 0):
        raise ValueError("y_true must not contain zero values for MAPE calculation")

    # Calculate absolute percentage errors
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true) * 100

    # Calculate mean of absolute percentage errors
    mape = np.mean(absolute_percentage_errors)

    return mape