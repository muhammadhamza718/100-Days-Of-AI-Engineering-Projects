"""
Linear regression implementation from scratch.

This module implements linear regression with gradient descent optimization
without relying on high-level libraries for core mathematical computations.
"""

import numpy as np
from typing import Tuple, Optional
from src.regression.base import BaseRegressor
from src.constants import EPSILON


class LinearRegressor(BaseRegressor):
    """
    Linear regression implementation using gradient descent.

    This class implements linear regression with the hypothesis function:
    hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
    And the cost function (MSE):
    J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the linear regressor.

        Args:
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
        """
        super().__init__(learning_rate, max_iterations, tolerance)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressor':
        """
        Fit the linear regression model to the training data using gradient descent.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            LinearRegressor: Fitted model instance
        """
        # Validate input dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples. "
                           f"Got X: {X.shape}, y: {y.shape}")

        # Store the number of samples and features
        n_samples, n_features = X.shape

        # Add intercept column (column of ones) to X
        X_with_intercept = self._add_intercept(X)

        # Initialize parameters (theta) with small random values
        self.theta = self._initialize_theta(n_features)

        # Initialize cost history for tracking convergence
        self.cost_history = []

        # Gradient descent algorithm
        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Calculate predictions using current parameters
            y_pred = self._compute_predictions(X_with_intercept)

            # Calculate the cost (Mean Squared Error)
            cost = self._compute_cost(X_with_intercept, y)
            self.cost_history.append(cost)

            # Calculate gradients
            gradients = self._compute_gradients(X_with_intercept, y, y_pred)

            # Update parameters using gradient descent rule
            self.theta -= self.learning_rate * gradients

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

            prev_cost = cost

        # Mark model as fitted
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted linear regression model.

        Args:
            X (np.ndarray): Features for prediction, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Add intercept column (column of ones) to X
        X_with_intercept = self._add_intercept(X)

        # Calculate predictions: hθ(x) = X * θ
        predictions = self._compute_predictions(X_with_intercept)

        return predictions

    def _compute_predictions(self, X_with_intercept: np.ndarray) -> np.ndarray:
        """
        Compute predictions using the current parameters.

        Args:
            X_with_intercept (np.ndarray): Feature matrix with intercept column

        Returns:
            np.ndarray: Predicted values
        """
        # Matrix multiplication: X * θ (dot product of each row with theta)
        predictions = X_with_intercept.dot(self.theta)
        return predictions

    def _compute_cost(self, X_with_intercept: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost function.

        The cost function is: J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

        Args:
            X_with_intercept (np.ndarray): Feature matrix with intercept column
            y (np.ndarray): True target values

        Returns:
            float: Cost value
        """
        # Calculate predictions
        predictions = self._compute_predictions(X_with_intercept)

        # Calculate errors
        errors = predictions - y

        # Calculate squared errors
        squared_errors = errors ** 2

        # Calculate mean squared error and multiply by 1/2
        cost = np.mean(squared_errors) / 2

        return cost

    def _compute_gradients(self, X_with_intercept: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function with respect to each parameter.

        The gradient for each parameter θⱼ is: ∂J/∂θⱼ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾

        Args:
            X_with_intercept (np.ndarray): Feature matrix with intercept column
            y (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values

        Returns:
            np.ndarray: Gradient vector for each parameter
        """
        # Calculate errors (predictions - actual values)
        errors = y_pred - y

        # Calculate gradients using matrix multiplication
        # Gradients = (1/m) * X.T * errors
        n_samples = X_with_intercept.shape[0]
        gradients = (1 / n_samples) * X_with_intercept.T.dot(errors)

        return gradients

    def get_parameters(self) -> Optional[np.ndarray]:
        """
        Get the learned parameters (theta) of the model.

        Returns:
            Optional[np.ndarray]: Model parameters if fitted, None otherwise
        """
        return self.theta

    def get_cost_history(self) -> Optional[list]:
        """
        Get the history of cost values during training.

        Returns:
            Optional[list]: List of cost values if model was trained, None otherwise
        """
        return self.cost_history


def compute_hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the hypothesis function hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        theta (np.ndarray): Parameter vector

    Returns:
        np.ndarray: Computed hypothesis values
    """
    return X.dot(theta)


def compute_mse_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the Mean Squared Error cost function.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector

    Returns:
        float: Cost value
    """
    predictions = compute_hypothesis(X, theta)
    errors = predictions - y
    squared_errors = errors ** 2
    cost = np.mean(squared_errors) / 2
    return cost


def compute_cost_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the cost function.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector

    Returns:
        np.ndarray: Gradient vector
    """
    predictions = compute_hypothesis(X, theta)
    errors = predictions - y
    n_samples = X.shape[0]
    gradients = (1 / n_samples) * X.T.dot(errors)
    return gradients