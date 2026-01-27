"""
Ridge Regression (L2 Regularization) implementation.

This module implements Ridge Regression which adds L2 regularization to linear regression
to prevent overfitting and handle multicollinearity.
"""

import numpy as np
from typing import Tuple, Optional
from src.regression.base import BaseRegressor
from src.regression.linear_regression import LinearRegressor
from src.constants import EPSILON


class RidgeRegressor(BaseRegressor):
    """
    Ridge Regression (L2 Regularized Linear Regression) implementation.

    Ridge regression adds an L2 penalty term to the cost function:
    J(θ) = MSE + λ * Σ(θⱼ²) for j ≠ 0 (excluding bias term)
    This shrinks coefficients towards zero but doesn't set them exactly to zero.
    """

    def __init__(self, lambda_reg: float = 0.01, learning_rate: float = 0.01,
                 max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the Ridge regressor.

        Args:
            lambda_reg (float): Regularization strength (λ)
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
        """
        super().__init__(learning_rate, max_iterations, tolerance)
        self.lambda_reg = lambda_reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressor':
        """
        Fit the Ridge regression model to the training data using gradient descent.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            RidgeRegressor: Fitted model instance
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

        # Ridge Regression with Gradient Descent
        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Calculate predictions using current parameters
            y_pred = self._compute_predictions(X_with_intercept)

            # Calculate the cost (MSE + L2 regularization)
            cost = self._compute_cost(X_with_intercept, y)
            self.cost_history.append(cost)

            # Calculate gradients with L2 regularization
            gradients = self._compute_gradients(X_with_intercept, y, y_pred)

            # Update parameters using gradient descent rule
            # For Ridge: θⱼ := θⱼ(1 - αλ/m) - α(1/m)Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾ for j ≠ 0
            # For j = 0 (bias): θ₀ := θ₀ - α(1/m)Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x₀⁽ⁱ⁾

            # Update bias term separately (no regularization)
            self.theta[0] -= self.learning_rate * gradients[0]

            # Update other parameters with regularization
            self.theta[1:] = self.theta[1:] * (1 - self.learning_rate * self.lambda_reg / n_samples) - \
                            self.learning_rate * gradients[1:]

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Ridge Regression converged after {iteration + 1} iterations")
                break

            prev_cost = cost

        # Mark model as fitted
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Ridge regression model.

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
        Compute the Ridge Regression cost function.

        The cost function is: J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/2m) Σ(θⱼ²) for j ≠ 0

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

        # Calculate mean squared error
        mse_cost = np.mean(squared_errors) / 2

        # Calculate L2 regularization term (excluding bias term at index 0)
        l2_penalty = (self.lambda_reg / (2 * len(y))) * np.sum(self.theta[1:] ** 2)

        # Total cost
        total_cost = mse_cost + l2_penalty

        return total_cost

    def _compute_gradients(self, X_with_intercept: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the Ridge Regression cost function with respect to each parameter.

        The gradient for each parameter θⱼ is: ∂J/∂θⱼ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾ + (λ/m)θⱼ for j ≠ 0
        For j = 0: ∂J/∂θ₀ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x₀⁽ⁱ⁾ (no regularization for bias)

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

        # Add L2 regularization term to gradients (excluding bias term at index 0)
        # For Ridge: gradient_j = gradient_j + (λ/m) * theta_j for j ≠ 0
        gradients[1:] += (self.lambda_reg / n_samples) * self.theta[1:]

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

    def get_regularization_strength(self) -> float:
        """
        Get the regularization strength (lambda) used in the model.

        Returns:
            float: Regularization strength
        """
        return self.lambda_reg


def compute_ridge_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_reg: float) -> float:
    """
    Compute the Ridge Regression cost function.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector
        lambda_reg (float): Regularization strength

    Returns:
        float: Ridge Regression cost value
    """
    # Calculate predictions
    predictions = X.dot(theta)

    # Calculate errors
    errors = predictions - y

    # Calculate squared errors
    squared_errors = errors ** 2

    # Calculate mean squared error
    mse_cost = np.mean(squared_errors) / 2

    # Calculate L2 regularization term (excluding bias term at index 0)
    l2_penalty = (lambda_reg / (2 * len(y))) * np.sum(theta[1:] ** 2)

    # Total cost
    total_cost = mse_cost + l2_penalty

    return total_cost


def compute_ridge_gradients(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients of the Ridge Regression cost function.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Gradient vector
    """
    # Calculate predictions
    predictions = X.dot(theta)

    # Calculate errors
    errors = predictions - y

    # Calculate gradients using matrix multiplication
    n_samples = X.shape[0]
    gradients = (1 / n_samples) * X.T.dot(errors)

    # Add L2 regularization term to gradients (excluding bias term at index 0)
    gradients[1:] += (lambda_reg / n_samples) * theta[1:]

    return gradients