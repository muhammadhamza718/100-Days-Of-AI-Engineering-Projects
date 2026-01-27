"""
Batch Gradient Descent optimizer implementation.

This module implements the batch gradient descent algorithm,
which uses the entire dataset to compute gradients at each iteration.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from src.constants import EPSILON


class BatchGradientDescent:
    """
    Batch Gradient Descent optimizer.

    Batch Gradient Descent computes the gradient using the entire training dataset
    at each iteration. This leads to stable convergence but can be slow for large datasets.
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize the Batch Gradient Descent optimizer.

        Args:
            learning_rate (float): Step size for parameter updates (α)
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence threshold for early stopping
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        self.converged = False

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 cost_func: Callable, grad_func: Callable,
                 initial_params: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Optimize parameters using Batch Gradient Descent.

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
        prev_cost = float('inf')

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Compute current cost
            cost = cost_func(params, X, y)
            self.cost_history.append(cost)

            # Compute gradients using the entire dataset
            gradients = grad_func(params, X, y)

            # Update parameters
            params -= self.learning_rate * gradients

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Batch GD converged after {iteration + 1} iterations")
                self.converged = True
                break

            prev_cost = cost

        return params, self.cost_history


def compute_cost_with_l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                                        cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L2 regularization term added.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cost_func (Callable): Base cost function
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L2 regularization added
    """
    # Compute base cost (e.g., MSE)
    base_cost = cost_func(params, X, y)

    # Compute L2 regularization term (excluding bias term at index 0)
    l2_penalty = (lambda_reg / (2 * len(y))) * np.sum(params[1:] ** 2)

    # Total cost
    total_cost = base_cost + l2_penalty

    return total_cost


def compute_cost_with_l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                                        cost_func: Callable, lambda_reg: float) -> float:
    """
    Compute cost with L1 regularization term added.

    Args:
        params (np.ndarray): Model parameters (excluding bias term at index 0)
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cost_func (Callable): Base cost function
        lambda_reg (float): Regularization strength

    Returns:
        float: Cost with L1 regularization added
    """
    # Compute base cost (e.g., MSE)
    base_cost = cost_func(params, X, y)

    # Compute L1 regularization term (excluding bias term at index 0)
    l1_penalty = (lambda_reg / len(y)) * np.sum(np.abs(params[1:]))

    # Total cost
    total_cost = base_cost + l1_penalty

    return total_cost


def compute_gradients_with_l2_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                                            grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L2 regularization term added.

    Args:
        params (np.ndarray): Model parameters
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        grad_func (Callable): Base gradient function
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Gradients with L2 regularization
    """
    # Compute base gradients
    base_gradients = grad_func(params, X, y)

    # Add L2 regularization term to gradients (excluding bias term at index 0)
    reg_term = np.zeros_like(params)
    reg_term[1:] = (lambda_reg / len(y)) * params[1:]

    # Total gradients
    total_gradients = base_gradients + reg_term

    return total_gradients


def compute_gradients_with_l1_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                                            grad_func: Callable, lambda_reg: float) -> np.ndarray:
    """
    Compute gradients with L1 regularization using subgradients.

    Args:
        params (np.ndarray): Model parameters
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        grad_func (Callable): Base gradient function
        lambda_reg (float): Regularization strength

    Returns:
        np.ndarray: Subgradients with L1 regularization
    """
    # Compute base gradients
    base_gradients = grad_func(params, X, y)

    # Add L1 regularization term using subgradients (excluding bias term at index 0)
    # For L1, the subgradient of |θⱼ| is sign(θⱼ) when θⱼ ≠ 0, and [-1,1] when θⱼ = 0
    l1_subgradient = np.zeros_like(params)
    l1_subgradient[1:] = (lambda_reg / len(y)) * np.sign(params[1:])

    # Total gradients
    total_gradients = base_gradients + l1_subgradient

    return total_gradients