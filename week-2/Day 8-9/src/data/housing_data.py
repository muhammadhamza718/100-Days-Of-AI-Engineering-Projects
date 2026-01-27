"""
Housing data utilities for the regression implementation.

This module provides utilities for loading and handling housing data,
including synthetic data generation for testing and validation purposes.
"""

import numpy as np
from typing import Tuple, Optional


def load_sample_housing_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a sample housing dataset for demonstration and testing purposes.

    This dataset simulates housing features and prices with realistic relationships
    to enable testing of the regression algorithms.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Features matrix with shape (n_samples, n_features)
              Features: [Square Footage, Number of Bedrooms, Age of House]
            - y (np.ndarray): Target values (prices) with shape (n_samples,)
    """
    # Sample housing data with realistic relationships
    np.random.seed(42)  # For reproducible results

    n_samples = 100

    # Generate synthetic housing features
    square_footage = np.random.uniform(800, 4000, size=n_samples)  # 800-4000 sq ft
    bedrooms = np.random.randint(1, 6, size=n_samples)            # 1-5 bedrooms
    age = np.random.uniform(0, 50, size=n_samples)                # 0-50 years old

    # Combine features into a single matrix
    X = np.column_stack([square_footage, bedrooms, age])

    # Generate prices based on features with some noise
    # Price = 100 * sqft + 15000 * bedrooms - 1000 * age + base_price + noise
    base_price = 50000
    price_per_sqft = 100
    price_per_bedroom = 15000
    depreciation_per_year = 1000

    y = (base_price +
         price_per_sqft * square_footage / 10 +
         price_per_bedroom * bedrooms -
         depreciation_per_year * age +
         np.random.normal(0, 10000, size=n_samples))  # Add noise

    return X, y


def load_custom_housing_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load housing data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing housing data

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Features matrix
            - y (np.ndarray): Target values (prices)
    """
    # This is a placeholder implementation
    # In a real implementation, we would read from CSV
    # For now, we'll return the sample data
    print(f"Loading custom data from {file_path} (using sample data for now)")
    return load_sample_housing_data()


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets.

    Args:
        X (np.ndarray): Features matrix
        y (np.ndarray): Target values
        test_size (float): Proportion of data to use for testing (between 0 and 1)
        random_state (Optional[int]): Random seed for reproducible splits

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_train (np.ndarray): Training features
            - X_test (np.ndarray): Testing features
            - y_train (np.ndarray): Training targets
            - y_test (np.ndarray): Testing targets
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Generate random indices for the test set
    test_indices = np.random.choice(n_samples, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def generate_nonlinear_housing_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a nonlinear housing dataset for testing polynomial regression.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Features matrix with shape (n_samples, n_features)
            - y (np.ndarray): Target values (prices) with shape (n_samples,)
    """
    np.random.seed(42)

    n_samples = 100

    # Generate features
    square_footage = np.random.uniform(800, 4000, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)

    # For nonlinear data, we'll use only square footage with quadratic relationship
    X = square_footage.reshape(-1, 1)

    # Quadratic relationship: price increases with sqft but at a decreasing rate
    y = 50000 + 100 * square_footage - 0.01 * square_footage**2 + np.random.normal(0, 5000, size=n_samples)

    return X, y


def add_noise_to_data(X: np.ndarray, y: np.ndarray, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add controlled noise to the dataset.

    Args:
        X (np.ndarray): Features matrix
        y (np.ndarray): Target values
        noise_level (float): Level of noise to add (as fraction of data range)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Dataset with added noise
    """
    X_noisy = X + np.random.normal(0, noise_level * np.std(X, axis=0), size=X.shape)
    y_noisy = y + np.random.normal(0, noise_level * np.std(y), size=y.shape)

    return X_noisy, y_noisy