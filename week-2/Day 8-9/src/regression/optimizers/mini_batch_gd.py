"""
Mini-Batch Gradient Descent optimizer implementation.

This module implements the mini-batch gradient descent algorithm,
which uses small random batches of samples to compute gradients at each iteration.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from src.constants import EPSILON


class MiniBatchGradientDescent:
    """
    Mini-Batch Gradient Descent optimizer.

    Mini-Batch Gradient Descent computes the gradient using small random batches
    of the training data at each iteration. This balances the stability of batch GD
    with the speed of stochastic GD.
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6,
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize the Mini-Batch Gradient Descent optimizer.

        Args:
            learning_rate (float): Step size for parameter updates (α)
            max_iterations (int): Maximum number of iterations (epochs through the dataset)
            tolerance (float): Convergence threshold (less applicable for mini-batch GD)
            batch_size (int): Size of each mini-batch
            shuffle (bool): Whether to shuffle the data at each epoch
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cost_history = []
        self.converged = False

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 cost_func: Callable, grad_func: Callable,
                 initial_params: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Optimize parameters using Mini-Batch Gradient Descent.

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

            # Process mini-batches
            epoch_cost = 0.0
            num_batches = (n_samples + self.batch_size - 1) // self.batch_size  # Ceiling division

            for batch_idx in range(num_batches):
                # Get mini-batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Compute gradients using mini-batch
                gradients = grad_func(params, X_batch, y_batch)

                # Update parameters
                params -= self.learning_rate * gradients

                # Compute and accumulate cost for monitoring
                batch_cost = cost_func(params, X_batch, y_batch)
                epoch_cost += batch_cost

            # Average cost for the epoch
            epoch_cost /= num_batches
            self.cost_history.append(epoch_cost)

        self.converged = True
        return params, self.cost_history


def compute_cost_with_l2_regularization_minibatch(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray,
                                                cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L2 regularization for a mini-batch in Mini-Batch GD.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        X_batch (np.ndarray): Mini-batch feature matrix
        y_batch (np.ndarray): Mini-batch target values
        cost_func (Callable): Base cost function for mini-batch
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L2 regularization for mini-batch
    """
    # Compute base cost for mini-batch
    base_cost = cost_func(params, X_batch, y_batch)

    # Compute L2 regularization term (excluding bias term at index 0)
    l2_penalty = (lambda_reg / (2 * len(y_batch))) * np.sum(params[1:] ** 2)

    # Total cost
    total_cost = base_cost + l2_penalty

    return total_cost


def compute_cost_with_l1_regularization_minibatch(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray,
                                                cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L1 regularization for a mini-batch in Mini-Batch GD.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        X_batch (np.ndarray): Mini-batch feature matrix
        y_batch (np.ndarray): Mini-batch target values
        cost_func (Callable): Base cost function for mini-batch
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L1 regularization for mini-batch
    """
    # Compute base cost for mini-batch
    base_cost = cost_func(params, X_batch, y_batch)

    # Compute L1 regularization term (excluding bias term at index 0)
    l1_penalty = (lambda_reg / len(y_batch)) * np.sum(np.abs(params[1:]))

    # Total cost
    total_cost = base_cost + l1_penalty

    return total_cost


def compute_gradients_with_l2_regularization_minibatch(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray,
                                                   grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L2 regularization for a mini-batch in Mini-Batch GD.

    Args:
        params (np.ndarray): Model parameters
        X_batch (np.ndarray): Mini-batch feature matrix
        y_batch (np.ndarray): Mini-batch target values
        grad_func (Callable): Base gradient function for mini-batch
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Gradients with L2 regularization for mini-batch
    """
    # Compute base gradients for mini-batch
    base_gradients = grad_func(params, X_batch, y_batch)

    # Add L2 regularization term to gradients (excluding bias term at index 0)
    reg_term = np.zeros_like(params)
    reg_term[1:] = (lambda_reg / len(y_batch)) * params[1:]

    # Total gradients
    total_gradients = base_gradients + reg_term

    return total_gradients


def compute_gradients_with_l1_regularization_minibatch(params: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray,
                                                   grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L1 regularization using subgradients for a mini-batch in Mini-Batch GD.

    Args:
        params (np.ndarray): Model parameters
        X_batch (np.ndarray): Mini-batch feature matrix
        y_batch (np.ndarray): Mini-batch target values
        grad_func (Callable): Base gradient function for mini-batch
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Subgradients with L1 regularization for mini-batch
    """
    # Compute base gradients for mini-batch
    base_gradients = grad_func(params, X_batch, y_batch)

    # Add L1 regularization term using subgradients (excluding bias term at index 0)
    l1_subgradient = np.zeros_like(params)
    l1_subgradient[1:] = (lambda_reg / len(y_batch)) * np.sign(params[1:])

    # Total gradients
    total_gradients = base_gradients + l1_subgradient

    return total_gradients


def create_mini_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list:
    """
    Create mini-batches from the dataset.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        batch_size (int): Size of each mini-batch

    Returns:
        list: List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        batches.append((X_batch, y_batch))

    return batches