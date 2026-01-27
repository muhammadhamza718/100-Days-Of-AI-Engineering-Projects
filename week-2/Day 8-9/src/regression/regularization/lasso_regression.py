"""
Lasso Regression (L1 Regularization) implementation.

This module implements Lasso Regression which adds L1 regularization to linear regression
to prevent overfitting and perform feature selection.
"""

import numpy as np
from typing import Tuple, Optional
from src.regression.base import BaseRegressor
from src.regression.linear_regression import LinearRegressor
from src.constants import EPSILON


class LassoRegressor(BaseRegressor):
    """
    Lasso Regression (L1 Regularized Linear Regression) implementation.

    Lasso regression adds an L1 penalty term to the cost function:
    J(θ) = MSE + λ * Σ|θⱼ| for j ≠ 0 (excluding bias term)
    This can set coefficients exactly to zero, performing feature selection.
    """

    def __init__(self, lambda_reg: float = 0.01, learning_rate: float = 0.01,
                 max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the Lasso regressor.

        Args:
            lambda_reg (float): Regularization strength (λ)
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
        """
        super().__init__(learning_rate, max_iterations, tolerance)
        self.lambda_reg = lambda_reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressor':
        """
        Fit the Lasso regression model to the training data using gradient descent.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            LassoRegressor: Fitted model instance
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

        # Lasso Regression with Gradient Descent (using subgradients for L1 penalty)
        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Calculate predictions using current parameters
            y_pred = self._compute_predictions(X_with_intercept)

            # Calculate the cost (MSE + L1 regularization)
            cost = self._compute_cost(X_with_intercept, y)
            self.cost_history.append(cost)

            # Calculate gradients with L1 regularization (subgradients)
            gradients = self._compute_gradients(X_with_intercept, y, y_pred)

            # Update parameters using gradient descent rule
            # For Lasso: θⱼ := θⱼ - α * sign(θⱼ) * λ/m - α(1/m)Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾ for j ≠ 0
            # For j = 0 (bias): θ₀ := θ₀ - α(1/m)Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x₀⁽ⁱ⁾

            # Update bias term separately (no regularization)
            self.theta[0] -= self.learning_rate * gradients[0]

            # Update other parameters with L1 regularization using soft thresholding
            # Apply subgradient update for L1 regularization
            for j in range(1, len(self.theta)):
                # Subgradient of |θⱼ| is sign(θⱼ) if θⱼ ≠ 0, and [-1,1] if θⱼ = 0
                # We use sign(θⱼ) as the subgradient
                subgradient_l1 = np.sign(self.theta[j])

                # Update with both MSE gradient and L1 subgradient
                self.theta[j] -= self.learning_rate * (
                    gradients[j] + (self.lambda_reg / n_samples) * subgradient_l1
                )

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Lasso Regression converged after {iteration + 1} iterations")
                break

            prev_cost = cost

        # Mark model as fitted
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Lasso regression model.

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
        Compute the Lasso Regression cost function.

        The cost function is: J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/m) Σ|θⱼ| for j ≠ 0

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

        # Calculate L1 regularization term (excluding bias term at index 0)
        l1_penalty = (self.lambda_reg / len(y)) * np.sum(np.abs(self.theta[1:]))

        # Total cost
        total_cost = mse_cost + l1_penalty

        return total_cost

    def _compute_gradients(self, X_with_intercept: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the Lasso Regression cost function with respect to each parameter.

        The gradient for each parameter θⱼ is: ∂J/∂θⱼ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾ + (λ/m)sign(θⱼ) for j ≠ 0
        For j = 0: ∂J/∂θ₀ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x₀⁽ⁱ⁾ (no regularization for bias)

        Args:
            X_with_intercept (np.ndarray): Feature matrix with intercept column
            y (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values

        Returns:
            np.ndarray: Gradient vector for each parameter (using subgradients for L1)
        """
        # Calculate errors (predictions - actual values)
        errors = y_pred - y

        # Calculate gradients using matrix multiplication for MSE part
        # Gradients = (1/m) * X.T * errors
        n_samples = X_with_intercept.shape[0]
        gradients = (1 / n_samples) * X_with_intercept.T.dot(errors)

        # Add L1 regularization term to gradients using subgradients (excluding bias term at index 0)
        # For Lasso: gradient_j = gradient_j + (λ/m) * sign(theta_j) for j ≠ 0
        # Note: sign(0) = 0, which is a valid subgradient of |θⱼ| at θⱼ = 0
        gradients[1:] += (self.lambda_reg / n_samples) * np.sign(self.theta[1:])

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

    def get_coefficient_sparsity(self) -> float:
        """
        Get the sparsity of the coefficient vector (fraction of zero coefficients).

        Returns:
            float: Fraction of coefficients that are exactly zero (excluding bias term)
        """
        if not self.is_fitted:
            return 0.0

        # Count non-zero coefficients (excluding bias term)
        non_zero_coeffs = np.count_nonzero(self.theta[1:])
        total_coeffs = len(self.theta) - 1  # Exclude bias term

        sparsity = 1.0 - (non_zero_coeffs / total_coeffs if total_coeffs > 0 else 0.0)
        return sparsity


def compute_lasso_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_reg: float) -> float:
    """
    Compute the Lasso Regression cost function.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector
        lambda_reg (float): Regularization strength

    Returns:
        float: Lasso Regression cost value
    """
    # Calculate predictions
    predictions = X.dot(theta)

    # Calculate errors
    errors = predictions - y

    # Calculate squared errors
    squared_errors = errors ** 2

    # Calculate mean squared error
    mse_cost = np.mean(squared_errors) / 2

    # Calculate L1 regularization term (excluding bias term at index 0)
    l1_penalty = (lambda_reg / len(y)) * np.sum(np.abs(theta[1:]))

    # Total cost
    total_cost = mse_cost + l1_penalty

    return total_cost


def compute_lasso_gradients(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients of the Lasso Regression cost function using subgradients.

    Args:
        X (np.ndarray): Feature matrix with intercept column
        y (np.ndarray): True target values
        theta (np.ndarray): Parameter vector
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Gradient vector (using subgradients for L1)
    """
    # Calculate predictions
    predictions = X.dot(theta)

    # Calculate errors
    errors = predictions - y

    # Calculate gradients using matrix multiplication for MSE part
    n_samples = X.shape[0]
    gradients = (1 / n_samples) * X.T.dot(errors)

    # Add L1 regularization term using subgradients (excluding bias term at index 0)
    gradients[1:] += (lambda_reg / n_samples) * np.sign(theta[1:])

    return gradients


def soft_threshold(x: float, threshold: float) -> float:
    """
    Apply soft thresholding operator.

    Soft thresholding is defined as: sign(x) * max(|x| - threshold, 0)

    Args:
        x (float): Input value
        threshold (float): Threshold value

    Returns:
        float: Soft thresholded value
    """
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0