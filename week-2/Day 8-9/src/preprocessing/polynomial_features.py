"""
Polynomial feature generation for regression implementation.

This module provides functions to generate polynomial features from input data,
which is essential for polynomial regression models.
"""

import numpy as np
from itertools import combinations_with_replacement
from typing import List, Tuple


def generate_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features up to the specified degree.

    This function creates polynomial combinations of the input features up to the given degree.
    For example, if X = [x1, x2] and degree = 2, the output will include:
    [1, x1, x2, x1^2, x1*x2, x2^2]

    Args:
        X (np.ndarray): Input features matrix of shape (n_samples, n_features)
        degree (int): Maximum degree of polynomial features to generate

    Returns:
        np.ndarray: Polynomial features matrix of shape (n_samples, n_output_features)
    """
    if degree < 1:
        raise ValueError(f"Degree must be at least 1, got {degree}")

    n_samples, n_features = X.shape

    # Generate all possible combinations of feature indices up to the given degree
    # This includes powers and interactions between features
    feature_combinations = []

    # Add bias term (intercept) as the first feature
    feature_combinations.append(np.ones((n_samples, 1)))

    # Generate polynomial terms for each degree from 1 to 'degree'
    for d in range(1, degree + 1):
        # Get all combinations with replacement of feature indices of length d
        for combo in combinations_with_replacement(range(n_features), d):
            # Calculate the product of the selected features for each sample
            term = np.ones(n_samples)
            for idx in combo:
                term *= X[:, idx]

            # Reshape to column vector and add to combinations
            feature_combinations.append(term.reshape(-1, 1))

    # Concatenate all feature combinations horizontally
    X_poly = np.concatenate(feature_combinations, axis=1)

    return X_poly


def generate_polynomial_features_manual(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features manually without using itertools.
    This is an alternative implementation that builds features iteratively.

    Args:
        X (np.ndarray): Input features matrix of shape (n_samples, n_features)
        degree (int): Maximum degree of polynomial features to generate

    Returns:
        np.ndarray: Polynomial features matrix of shape (n_samples, n_output_features)
    """
    if degree < 1:
        raise ValueError(f"Degree must be at least 1, got {degree}")

    n_samples, n_features = X.shape

    # Start with bias term (intercept)
    poly_features = [np.ones(n_samples)]

    # Add original features (degree 1)
    for i in range(n_features):
        poly_features.append(X[:, i])

    # Generate higher degree terms iteratively
    for d in range(2, degree + 1):
        # For each degree d, generate all combinations of features with that degree
        # This is done by multiplying existing features of lower degrees

        # Generate all possible ways to distribute degree 'd' among 'n_features'
        # For example, for 2 features and degree 2: (2,0), (1,1), (0,2)
        degree_combinations = _generate_degree_combinations(n_features, d)

        for deg_combo in degree_combinations:
            # Calculate the product of features raised to their respective powers
            term = np.ones(n_samples)
            for feat_idx, power in enumerate(deg_combo):
                if power > 0:
                    term *= X[:, feat_idx] ** power
            poly_features.append(term)

    # Stack all features as columns
    X_poly = np.column_stack(poly_features)

    return X_poly


def _generate_degree_combinations(n_features: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Generate all possible combinations of degrees that sum to max_degree.

    Args:
        n_features (int): Number of input features
        max_degree (int): Maximum degree for the combination

    Returns:
        List[Tuple[int, ...]]: List of tuples representing degree combinations
    """
    combinations = []

    # Use recursive approach to generate all combinations
    def generate_recursive(current_combo, remaining_degree, remaining_features):
        if remaining_features == 1:
            # Last feature gets all remaining degree
            combinations.append(tuple(current_combo + [remaining_degree]))
        elif remaining_degree == 0:
            # No more degree to distribute, pad with zeros
            combinations.append(tuple(current_combo + [0] * remaining_features))
        else:
            # Try all possible degrees for the current feature
            for degree in range(remaining_degree + 1):
                generate_recursive(
                    current_combo + [degree],
                    remaining_degree - degree,
                    remaining_features - 1
                )

    generate_recursive([], max_degree, n_features)
    return combinations


def count_polynomial_features(n_features: int, degree: int) -> int:
    """
    Calculate the number of polynomial features that will be generated.

    This is useful for memory allocation and complexity estimation.

    Args:
        n_features (int): Number of input features
        degree (int): Maximum degree of polynomial features

    Returns:
        int: Number of polynomial features that will be generated
    """
    # The number of polynomial features is the number of ways to choose
    # degree elements with repetition from n_features + 1 (including bias term)
    # This is equivalent to (n_features + degree) choose degree
    import math

    # Calculate binomial coefficient (n_features + degree) choose degree
    numerator = math.factorial(n_features + degree)
    denominator = math.factorial(n_features) * math.factorial(degree)

    return numerator // denominator


def polynomial_feature_names(feature_names: List[str], degree: int = 2) -> List[str]:
    """
    Generate names for polynomial features based on original feature names.

    Args:
        feature_names (List[str]): Names of the original features
        degree (int): Maximum degree of polynomial features

    Returns:
        List[str]: Names for the polynomial features
    """
    if degree < 1:
        raise ValueError(f"Degree must be at least 1, got {degree}")

    names = ['bias']  # Start with bias term

    # Add original features (degree 1)
    names.extend(feature_names)

    # Generate higher degree terms
    for d in range(2, degree + 1):
        # Generate all combinations with replacement of feature indices of length d
        for combo in combinations_with_replacement(range(len(feature_names)), d):
            # Create name by joining the feature names with '*'
            name_parts = [feature_names[idx] for idx in combo]
            names.append('*'.join(name_parts))

    return names


class PolynomialFeatures:
    """
    A class to encapsulate polynomial feature generation with fitted parameters.
    """

    def __init__(self, degree: int = 2):
        """
        Initialize the polynomial features transformer.

        Args:
            degree (int): Maximum degree of polynomial features to generate
        """
        self.degree = degree
        self.n_input_features = None
        self.n_output_features = None

    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Fit the polynomial features transformer to the data.

        Args:
            X (np.ndarray): Input features matrix

        Returns:
            PolynomialFeatures: Fitted transformer instance
        """
        self.n_input_features = X.shape[1]
        self.n_output_features = count_polynomial_features(self.n_input_features, self.degree)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input features to polynomial features.

        Args:
            X (np.ndarray): Input features matrix to transform

        Returns:
            np.ndarray: Transformed polynomial features matrix
        """
        if self.n_input_features is None:
            raise RuntimeError("Transformer must be fitted before transforming data")

        if X.shape[1] != self.n_input_features:
            raise ValueError(f"Expected {self.n_input_features} features, got {X.shape[1]}")

        return generate_polynomial_features(X, self.degree)

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, 'PolynomialFeatures']:
        """
        Fit the transformer and transform the data in one step.

        Args:
            X (np.ndarray): Input features matrix to fit and transform

        Returns:
            Tuple[np.ndarray, PolynomialFeatures]: Transformed features and fitted transformer
        """
        self.fit(X)
        return self.transform(X), self