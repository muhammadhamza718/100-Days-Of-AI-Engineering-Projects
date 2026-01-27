"""
Base regressor class implementing the fundamental interface for regression models.

This class defines the common interface and shared functionality for all regression
models in the system. It enforces the implementation of core methods while providing
common utilities for regression tasks.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseRegressor(ABC):
    """
    Abstract base class for all regression models.

    This class provides the fundamental interface that all regression models must implement,
    ensuring consistency across different regression algorithms (Linear, Polynomial, Ridge, Lasso).
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the regressor with common hyperparameters.

        Args:
            learning_rate (float): Step size for gradient descent (α)
            max_iterations (int): Maximum number of training iterations
            tolerance (float): Convergence threshold for early stopping
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Model parameters to be learned
        self.theta: Optional[np.ndarray] = None
        self.cost_history: Optional[list] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegressor':
        """
        Fit the regression model to the training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)

        Returns:
            BaseRegressor: Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X (np.ndarray): Features for prediction, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values, shape (n_samples,)
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² coefficient of determination.

        Args:
            X (np.ndarray): Features, shape (n_samples, n_features)
            y (np.ndarray): True target values, shape (n_samples,)

        Returns:
            float: R² score indicating the proportion of variance explained
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        y_pred = self.predict(X)

        # Calculate total sum of squares
        ss_total = np.sum((y - np.mean(y)) ** 2)

        # Calculate residual sum of squares
        ss_residual = np.sum((y - y_pred) ** 2)

        # Calculate R² score
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

        return r_squared

    def _initialize_theta(self, n_features: int) -> np.ndarray:
        """
        Initialize model parameters (theta) with small random values.

        Args:
            n_features (int): Number of features in the dataset

        Returns:
            np.ndarray: Initialized parameters array of shape (n_features + 1,)
        """
        # Add 1 for the intercept term (bias)
        theta = np.random.normal(0, 0.01, size=(n_features + 1,))
        return theta

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add intercept column (column of ones) to feature matrix.

        Args:
            X (np.ndarray): Original feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Feature matrix with intercept column, shape (n_samples, n_features + 1)
        """
        # Add a column of ones for the intercept/bias term
        intercept = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate([intercept, X], axis=1)
        return X_with_intercept