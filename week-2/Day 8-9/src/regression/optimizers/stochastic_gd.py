"""
Stochastic Gradient Descent optimizer implementation.

This module implements the stochastic gradient descent algorithm,
which uses a single sample to compute gradients at each iteration.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from src.constants import EPSILON


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent optimizer.

    Stochastic Gradient Descent computes the gradient using a single randomly selected
    training example at each iteration. This leads to faster updates but more noisy convergence.
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6,
                 shuffle: bool = True):
        """
        Initialize the Stochastic Gradient Descent optimizer.

        Args:
            learning_rate (float): Step size for parameter updates (α)
            max_iterations (int): Maximum number of iterations (epochs through the dataset)
            tolerance (float): Convergence threshold (less applicable for SGD)
            shuffle (bool): Whether to shuffle the data at each epoch
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.shuffle = shuffle
        self.cost_history = []
        self.converged = False

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 cost_func: Callable, grad_func: Callable,
                 initial_params: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Optimize parameters using Stochastic Gradient Descent.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            cost_func (Callable): Function that computes cost given params, X, y
            grad_func (Callable): Function that computes gradients given params, X, y
            initial_params (np.ndarray): Initial parameter values

        Returns:
            Tuple[np.ndarray, list]: Optimized parameters and cost history
        """
        # Initialize parameters
        params = initial_params.copy()

        # Initialize cost history
        self.cost_history = []

        n_samples = X.shape[0]

        # Main optimization loop (iterate over epochs)
        for epoch in range(self.max_iterations):
            # Shuffle the data at each epoch if requested
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            # Iterate over each sample in the shuffled dataset
            epoch_cost = 0.0
            for i in range(n_samples):
                # Get single sample
                x_i = X_shuffled[i:i+1, :]  # Keep as 2D array
                y_i = y_shuffled[i]

                # Compute gradients using single sample
                gradients = grad_func(params, x_i, np.array([y_i]))

                # Update parameters
                params -= self.learning_rate * gradients

                # Compute and accumulate cost for monitoring
                sample_cost = cost_func(params, x_i, np.array([y_i]))
                epoch_cost += sample_cost

            # Average cost for the epoch
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)

        self.converged = True
        return params, self.cost_history


def compute_cost_with_l2_regularization_sgd(params: np.ndarray, x_single: np.ndarray, y_single: np.ndarray,
                                          cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L2 regularization for a single sample in SGD.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        x_single (np.ndarray): Single feature vector
        y_single (np.ndarray): Single target value
        cost_func (Callable): Base cost function for single sample
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L2 regularization for single sample
    """
    # Compute base cost for single sample
    base_cost = cost_func(params, x_single, y_single)

    # Compute L2 regularization term (excluding bias term at index 0)
    l2_penalty = (lambda_reg / 2) * np.sum(params[1:] ** 2)

    # Total cost
    total_cost = base_cost + l2_penalty

    return total_cost


def compute_cost_with_l1_regularization_sgd(params: np.ndarray, x_single: np.ndarray, y_single: np.ndarray,
                                          cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L1 regularization for a single sample in SGD.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        x_single (np.ndarray): Single feature vector
        y_single (np.ndarray): Single target value
        cost_func (Callable): Base cost function for single sample
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L1 regularization for single sample
    """
    # Compute base cost for single sample
    base_cost = cost_func(params, x_single, y_single)

    # Compute L1 regularization term (excluding bias term at index 0)
    l1_penalty = lambda_reg * np.sum(np.abs(params[1:]))

    # Total cost
    total_cost = base_cost + l1_penalty

    return total_cost


def compute_gradients_with_l2_regularization_sgd(params: np.ndarray, x_single: np.ndarray, y_single: np.ndarray,
                                              grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L2 regularization for a single sample in SGD.

    Args:
        params (np.ndarray): Model parameters
        x_single (np.ndarray): Single feature vector
        y_single (np.ndarray): Single target value
        grad_func (Callable): Base gradient function for single sample
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Gradients with L2 regularization for single sample
    """
    # Compute base gradients for single sample
    base_gradients = grad_func(params, x_single, y_single)

    # Add L2 regularization term to gradients (excluding bias term at index 0)
    reg_term = np.zeros_like(params)
    reg_term[1:] = lambda_reg * params[1:]

    # Total gradients
    total_gradients = base_gradients + reg_term

    return total_gradients


def compute_gradients_with_l1_regularization_sgd(params: np.ndarray, x_single: np.ndarray, y_single: np.ndarray,
                                              grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L1 regularization using subgradients for a single sample in SGD.

    Args:
        params (np.ndarray): Model parameters
        x_single (np.ndarray): Single feature vector
        y_single (np.ndarray): Single target value
        grad_func (Callable): Base gradient function for single sample
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Subgradients with L1 regularization for single sample
    """
    # Compute base gradients for single sample
    base_gradients = grad_func(params, x_single, y_single)

    # Add L1 regularization term using subgradients (excluding bias term at index 0)
    l1_subgradient = np.zeros_like(params)
    l1_subgradient[1:] = lambda_reg * np.sign(params[1:])

    # Total gradients
    total_gradients = base_gradients + l1_subgradient

    return total_gradients