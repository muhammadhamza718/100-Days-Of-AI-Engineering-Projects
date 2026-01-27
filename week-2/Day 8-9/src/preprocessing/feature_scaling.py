"""
Feature scaling utilities for regression implementation.

This module provides functions to normalize and standardize features,
which is crucial for gradient descent convergence in regression models.
"""

import numpy as np
from typing import Tuple, Optional
from src.utils.matrix_ops import mean_axis, std_axis


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to have values in the range [0, 1].

    This scaling method subtracts the minimum value and divides by the range
    (max - min) for each feature, resulting in values between 0 and 1.

    Args:
        X (np.ndarray): Input features matrix of shape (n_samples, n_features)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_normalized (np.ndarray): Normalized features matrix
            - min_vals (np.ndarray): Minimum values used for normalization
            - max_vals (np.ndarray): Maximum values used for normalization
    """
    # Calculate min and max for each feature (column)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Calculate range (max - min) for each feature
    ranges = max_vals - min_vals

    # Handle case where a feature has constant value (range = 0)
    # In this case, set range to 1 to avoid division by zero
    ranges = np.where(ranges == 0, 1, ranges)

    # Normalize: (X - min) / (max - min)
    X_normalized = (X - min_vals) / ranges

    return X_normalized, min_vals, max_vals


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to have mean=0 and std=1 (Z-score normalization).

    This scaling method subtracts the mean and divides by the standard deviation
    for each feature, resulting in standardized values.

    Args:
        X (np.ndarray): Input features matrix of shape (n_samples, n_features)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_standardized (np.ndarray): Standardized features matrix
            - means (np.ndarray): Means used for standardization
            - stds (np.ndarray): Standard deviations used for standardization
    """
    # Calculate mean and std for each feature (column)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Handle case where a feature has constant value (std = 0)
    # In this case, set std to 1 to avoid division by zero
    stds = np.where(stds == 0, 1, stds)

    # Standardize: (X - mean) / std
    X_standardized = (X - means) / stds

    return X_standardized, means, stds


def apply_normalization(X: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Apply previously calculated normalization parameters to new data.

    Args:
        X (np.ndarray): Input features matrix to normalize
        min_vals (np.ndarray): Minimum values from training data
        max_vals (np.ndarray): Maximum values from training data

    Returns:
        np.ndarray: Normalized features matrix
    """
    # Calculate range (max - min) for each feature
    ranges = max_vals - min_vals

    # Handle case where a feature has constant value (range = 0)
    ranges = np.where(ranges == 0, 1, ranges)

    # Apply normalization: (X - min) / (max - min)
    X_normalized = (X - min_vals) / ranges

    return X_normalized


def apply_standardization(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Apply previously calculated standardization parameters to new data.

    Args:
        X (np.ndarray): Input features matrix to standardize
        means (np.ndarray): Means from training data
        stds (np.ndarray): Standard deviations from training data

    Returns:
        np.ndarray: Standardized features matrix
    """
    # Handle case where a feature has constant value (std = 0)
    stds = np.where(stds == 0, 1, stds)

    # Apply standardization: (X - mean) / std
    X_standardized = (X - means) / stds

    return X_standardized


def denormalize_features(X_normalized: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Convert normalized features back to original scale.

    Args:
        X_normalized (np.ndarray): Normalized features matrix
        min_vals (np.ndarray): Minimum values used for normalization
        max_vals (np.ndarray): Maximum values used for normalization

    Returns:
        np.ndarray: Features matrix in original scale
    """
    # Calculate range (max - min) for each feature
    ranges = max_vals - min_vals

    # Handle case where a feature has constant value (range = 0)
    ranges = np.where(ranges == 0, 1, ranges)

    # Denormalize: X_original = X_normalized * (max - min) + min
    X_original = X_normalized * ranges + min_vals

    return X_original


def destandardize_features(X_standardized: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Convert standardized features back to original scale.

    Args:
        X_standardized (np.ndarray): Standardized features matrix
        means (np.ndarray): Means used for standardization
        stds (np.ndarray): Standard deviations used for standardization

    Returns:
        np.ndarray: Features matrix in original scale
    """
    # Handle case where a feature has constant value (std = 0)
    stds = np.where(stds == 0, 1, stds)

    # Destandardize: X_original = X_standardized * std + mean
    X_original = X_standardized * stds + means

    return X_original


class FeatureScaler:
    """
    A class to encapsulate feature scaling operations with fitted parameters.
    """

    def __init__(self, method: str = 'standardize'):
        """
        Initialize the scaler.

        Args:
            method (str): Scaling method ('standardize' or 'normalize')
        """
        self.method = method
        self.fitted = False
        self.scaling_params = {}

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        Fit the scaler to the training data.

        Args:
            X (np.ndarray): Training features matrix

        Returns:
            FeatureScaler: Fitted scaler instance
        """
        if self.method == 'standardize':
            _, means, stds = standardize_features(X)
            self.scaling_params = {'means': means, 'stds': stds}
        elif self.method == 'normalize':
            _, min_vals, max_vals = normalize_features(X)
            self.scaling_params = {'min_vals': min_vals, 'max_vals': max_vals}
        else:
            raise ValueError(f"Unknown scaling method: {self.method}. Use 'standardize' or 'normalize'.")

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.

        Args:
            X (np.ndarray): Features matrix to transform

        Returns:
            np.ndarray: Transformed features matrix
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transforming data")

        if self.method == 'standardize':
            return apply_standardization(X, self.scaling_params['means'], self.scaling_params['stds'])
        elif self.method == 'normalize':
            return apply_normalization(X, self.scaling_params['min_vals'], self.scaling_params['max_vals'])

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, 'FeatureScaler']:
        """
        Fit the scaler and transform the data in one step.

        Args:
            X (np.ndarray): Features matrix to fit and transform

        Returns:
            Tuple[np.ndarray, FeatureScaler]: Transformed features and fitted scaler
        """
        self.fit(X)
        return self.transform(X), self

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled features back to original scale.

        Args:
            X_scaled (np.ndarray): Scaled features matrix

        Returns:
            np.ndarray: Features matrix in original scale
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse transforming data")

        if self.method == 'standardize':
            return destandardize_features(X_scaled, self.scaling_params['means'], self.scaling_params['stds'])
        elif self.method == 'normalize':
            return denormalize_features(X_scaled, self.scaling_params['min_vals'], self.scaling_params['max_vals'])