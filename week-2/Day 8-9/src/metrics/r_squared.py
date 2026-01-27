"""
R-squared and related metrics implementation.

This module provides functions to calculate R-squared (coefficient of determination)
and related metrics for evaluating regression model performance.
"""

import numpy as np
from typing import Tuple
from src.metrics.mse import calculate_mse


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R-squared (coefficient of determination) score.

    R² is calculated as: R² = 1 - (SS_res / SS_tot)
    Where SS_res is the residual sum of squares and SS_tot is the total sum of squares.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: R-squared score (ranges from -∞ to 1.0, where 1.0 is perfect prediction)
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate the mean of true values
    y_mean = np.mean(y_true)

    # Calculate total sum of squares (variance in true values)
    ss_total = np.sum((y_true - y_mean) ** 2)

    # Calculate residual sum of squares (variance not explained by model)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Calculate R-squared
    if ss_total == 0:
        # If all true values are the same, return 1.0 if predictions are also the same
        # Otherwise return 0.0
        return 1.0 if ss_residual == 0 else 0.0

    r_squared = 1 - (ss_residual / ss_total)

    return r_squared


def calculate_adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate the adjusted R-squared score.

    Adjusted R² adjusts for the number of predictors in the model:
    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
    Where n is the number of observations and k is the number of features.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        n_features (int): Number of features in the model

    Returns:
        float: Adjusted R-squared score
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    n_observations = len(y_true)

    # Calculate regular R-squared
    r_squared = calculate_r_squared(y_true, y_pred)

    # Calculate adjusted R-squared
    if n_observations <= n_features + 1:
        # If we have more features than observations, return negative infinity
        return float('-inf')

    adjusted_r_squared = 1 - ((1 - r_squared) * (n_observations - 1) / (n_observations - n_features - 1))

    return adjusted_r_squared


def calculate_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between true and predicted values.

    The correlation coefficient measures the linear relationship between two variables.
    It ranges from -1 to 1, where 1 indicates a perfect positive linear relationship,
    -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: Pearson correlation coefficient
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Calculate numerator (covariance)
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))

    # Calculate denominators (standard deviations)
    sum_sq_true = np.sum((y_true - mean_true) ** 2)
    sum_sq_pred = np.sum((y_pred - mean_pred) ** 2)

    # Calculate correlation coefficient
    denominator = np.sqrt(sum_sq_true * sum_sq_pred)

    if denominator == 0:
        # If one of the variables has zero variance, correlation is undefined
        return 0.0

    correlation = numerator / denominator

    return correlation


def calculate_metrics_summary(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = None) -> dict:
    """
    Calculate a comprehensive set of regression metrics.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        n_features (int, optional): Number of features for adjusted R-squared calculation

    Returns:
        dict: Dictionary containing various regression metrics
    """
    # Validate input arrays
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # Calculate individual metrics
    mse = calculate_mse(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r_squared = calculate_r_squared(y_true, y_pred)
    correlation = calculate_correlation_coefficient(y_true, y_pred)

    # Calculate adjusted R-squared if number of features is provided
    if n_features is not None:
        adj_r_squared = calculate_adjusted_r_squared(y_true, y_pred, n_features)
    else:
        adj_r_squared = None

    # Calculate mean absolute percentage error (avoiding division by zero)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'adjusted_r_squared': adj_r_squared,
        'correlation_coefficient': correlation,
        'mape': mape,
        'n_observations': len(y_true),
        'n_features': n_features
    }


def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for the mean prediction error.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval
    """
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore', category=ImportWarning)

    # Calculate residuals (prediction errors)
    residuals = y_true - y_pred

    # Calculate sample statistics
    n = len(residuals)
    mean_error = np.mean(residuals)
    std_error = np.std(residuals, ddof=1)  # Use sample standard deviation

    # Calculate t-value for the given confidence level
    # For large samples, this approximates the z-value
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1) if n > 30 else 1.96  # Use z-value approximation for large samples

    # Calculate margin of error
    margin_of_error = t_value * (std_error / np.sqrt(n))

    # Calculate confidence interval
    lower_bound = mean_error - margin_of_error
    upper_bound = mean_error + margin_of_error

    return lower_bound, upper_bound


def calculate_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals for individual predictions.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the prediction intervals
    """
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore', category=ImportWarning)

    # Calculate residuals (prediction errors)
    residuals = y_true - y_pred

    # Calculate residual standard error (standard deviation of residuals)
    n = len(residuals)
    residual_std = np.sqrt(np.sum(residuals ** 2) / (n - 2))  # Assuming simple regression

    # Calculate t-value for the given confidence level
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-2) if n > 30 else 1.96  # Use z-value approximation for large samples

    # Calculate margin of error for prediction intervals
    # Note: For prediction intervals, we use a factor that accounts for the uncertainty
    # in both the mean prediction and the variability of individual observations
    margin_of_error = t_value * residual_std

    # Calculate prediction intervals
    lower_bounds = y_pred - margin_of_error
    upper_bounds = y_pred + margin_of_error

    return lower_bounds, upper_bounds