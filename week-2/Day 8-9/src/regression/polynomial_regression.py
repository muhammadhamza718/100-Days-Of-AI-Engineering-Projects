"""
Polynomial Regression implementation.

This module extends linear regression to support polynomial features,
allowing for modeling of non-linear relationships between features and target.
"""

import numpy as np
from typing import Tuple, Optional
from src.regression.linear_regression import LinearRegressor
from src.preprocessing.polynomial_features import generate_polynomial_features, PolynomialFeatures
from src.constants import EPSILON


class PolynomialRegressor(LinearRegressor):
    """
    Polynomial Regression implementation.

    This class extends LinearRegressor to support polynomial features.
    It automatically transforms input features to polynomial form before applying linear regression.
    """

    def __init__(self, degree: int = 2, learning_rate: float = 0.01,
                 max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the Polynomial regressor.

        Args:
            degree (int): Degree of polynomial features to generate
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
        """
        super().__init__(learning_rate, max_iterations, tolerance)
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.original_n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegressor':
        """
        Fit the polynomial regression model to the training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            PolynomialRegressor: Fitted model instance
        """
        # Store the number of original features
        self.original_n_features = X.shape[1]

        # Transform features to polynomial form
        X_poly, _ = self.poly_features.fit_transform(X)

        # Call parent fit method with polynomial features
        super().fit(X_poly, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted polynomial regression model.

        Args:
            X (np.ndarray): Features for prediction, shape (n_samples, n_original_features)

        Returns:
            np.ndarray: Predicted values, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if X.shape[1] != self.original_n_features:
            raise ValueError(f"Expected {self.original_n_features} features, got {X.shape[1]}")

        # Transform features to polynomial form using fitted parameters
        X_poly = self.poly_features.transform(X)

        # Add intercept column (column of ones) to polynomial features
        X_poly_with_intercept = self._add_intercept(X_poly)

        # Calculate predictions: hθ(x) = X_poly * θ
        predictions = self._compute_predictions(X_poly_with_intercept)

        return predictions

    def _compute_predictions(self, X_poly_with_intercept: np.ndarray) -> np.ndarray:
        """
        Compute predictions using the current parameters.

        Args:
            X_poly_with_intercept (np.ndarray): Polynomial feature matrix with intercept column

        Returns:
            np.ndarray: Predicted values
        """
        # Matrix multiplication: X_poly * θ (dot product of each row with theta)
        predictions = X_poly_with_intercept.dot(self.theta)
        return predictions

    def get_polynomial_degree(self) -> int:
        """
        Get the degree of polynomial features used in the model.

        Returns:
            int: Polynomial degree
        """
        return self.degree

    def get_original_n_features(self) -> Optional[int]:
        """
        Get the number of original features (before polynomial transformation).

        Returns:
            Optional[int]: Number of original features if fitted, None otherwise
        """
        return self.original_n_features


def create_polynomial_features_manual(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Manually create polynomial features up to the specified degree.

    Args:
        X (np.ndarray): Input features matrix of shape (n_samples, n_features)
        degree (int): Maximum degree of polynomial features to generate

    Returns:
        np.ndarray: Polynomial features matrix of shape (n_samples, n_output_features)
    """
    n_samples, n_features = X.shape

    # Start with bias term (intercept)
    poly_features = [np.ones(n_samples)]

    # Add original features (degree 1)
    for i in range(n_features):
        poly_features.append(X[:, i])

    # Generate higher degree terms
    for d in range(2, degree + 1):
        # For each degree d, generate all combinations of features with that degree
        import itertools

        # Generate all possible ways to distribute degree 'd' among 'n_features'
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            # Calculate the product of features raised to their respective powers
            term = np.ones(n_samples)
            for idx in combo:
                term *= X[:, idx]
            poly_features.append(term)

    # Stack all features as columns
    X_poly = np.column_stack(poly_features)

    return X_poly


class AdaptivePolynomialRegressor(PolynomialRegressor):
    """
    Adaptive Polynomial Regression that can dynamically adjust the polynomial degree
    based on model performance or other criteria.
    """

    def __init__(self, max_degree: int = 5, learning_rate: float = 0.01,
                 max_iterations: int = 1000, tolerance: float = 1e-6, cv_folds: int = 5):
        """
        Initialize the Adaptive Polynomial regressor.

        Args:
            max_degree (int): Maximum degree to consider during adaptation
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
            cv_folds (int): Number of folds for cross-validation
        """
        super().__init__(degree=1, learning_rate=learning_rate,
                         max_iterations=max_iterations, tolerance=tolerance)
        self.max_degree = max_degree
        self.cv_folds = cv_folds
        self.best_degree = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptivePolynomialRegressor':
        """
        Fit the adaptive polynomial regression model by selecting the best degree.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            AdaptivePolynomialRegressor: Fitted model instance
        """
        best_score = float('-inf')
        best_model_params = None

        # Try different polynomial degrees
        for degree in range(1, self.max_degree + 1):
            # Create a temporary model with current degree
            temp_model = PolynomialRegressor(
                degree=degree,
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )

            # Perform cross-validation to evaluate this degree
            cv_score = self._cross_validate(temp_model, X, y)

            if cv_score > best_score:
                best_score = cv_score
                self.best_degree = degree
                best_model_params = temp_model.get_parameters()

        # Now fit the final model with the best degree
        self.degree = self.best_degree
        self.poly_features = PolynomialFeatures(degree=self.best_degree)

        # Store the number of original features
        self.original_n_features = X.shape[1]

        # Transform features to polynomial form with the best degree
        X_poly, _ = self.poly_features.fit_transform(X)

        # Call parent fit method with polynomial features
        super(LinearRegressor, self).fit(X_poly, y)

        return self

    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform cross-validation to evaluate model performance.

        Args:
            model: Model to evaluate
            X (np.ndarray): Features matrix
            y (np.ndarray): Target values

        Returns:
            float: Cross-validation score (average R² score across folds)
        """
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_folds
        scores = []

        for fold in range(self.cv_folds):
            # Define validation indices
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.cv_folds - 1 else n_samples

            # Split data
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            X_train = np.vstack([X[:start_idx], X[end_idx:]])
            y_train = np.hstack([y[:start_idx], y[end_idx:]])

            # Fit model on training data
            model.fit(X_train, y_train)

            # Evaluate on validation data
            score = model.score(X_val, y_val)
            scores.append(score)

        # Return average score
        return np.mean(scores)