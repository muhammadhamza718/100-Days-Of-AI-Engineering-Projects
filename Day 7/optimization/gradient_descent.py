"""
Gradient Descent Implementation from Scratch

This module implements the gradient descent optimization algorithm from scratch,
following the mathematical foundations of machine learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


def gradient_descent(
    objective_fn: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    start_point: np.ndarray,
    learning_rate: float,
    num_iterations: int
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Implements the gradient descent algorithm from scratch.

    Math formula:
    θ_{t+1} = θ_t - α * ∇f(θ_t)

    Where:
    - θ_t is the parameter vector at iteration t
    - α is the learning rate
    - ∇f(θ_t) is the gradient of the objective function

    Args:
        objective_fn: Function to minimize
        gradient_fn: Gradient of the objective function
        start_point: Initial parameter values
        learning_rate: Step size for each iteration
        num_iterations: Number of iterations to perform

    Returns:
        Tuple of (path of parameter values, path of objective function values)
    """
    current_point = start_point.copy()
    path_points = [current_point.copy()]
    path_values = [objective_fn(current_point)]

    for i in range(num_iterations):
        grad = gradient_fn(current_point)
        current_point = current_point - learning_rate * grad

        path_points.append(current_point.copy())
        path_values.append(objective_fn(current_point))

    return path_points, path_values


def quadratic_function(params: np.ndarray) -> float:
    """
    Quadratic function: f(x,y) = x² + y²

    Args:
        params: Array with [x, y] values

    Returns:
        Function value at the given point
    """
    return np.sum(params ** 2)


def quadratic_gradient(params: np.ndarray) -> np.ndarray:
    """
    Gradient of quadratic function: ∇f(x,y) = [2x, 2y]

    Args:
        params: Array with [x, y] values

    Returns:
        Gradient vector at the given point
    """
    return 2 * params


def rosenbrock_function(params: np.ndarray, a: float = 1, b: float = 100) -> float:
    """
    Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²

    Args:
        params: Array with [x, y] values
        a: Parameter a (default 1)
        b: Parameter b (default 100)

    Returns:
        Function value at the given point
    """
    x, y = params[0], params[1]
    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_gradient(params: np.ndarray, a: float = 1, b: float = 100) -> np.ndarray:
    """
    Gradient of Rosenbrock function:
    ∂f/∂x = -2(a-x) - 4bx(y-x²)
    ∂f/∂y = 2b(y-x²)

    Args:
        params: Array with [x, y] values
        a: Parameter a (default 1)
        b: Parameter b (default 100)

    Returns:
        Gradient vector at the given point
    """
    x, y = params[0], params[1]
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])


def visualize_convergence(values: List[float], title: str = "Convergence Curve", save_path: str = None):
    """
    Plot the convergence curve of the optimization process.

    Args:
        values: List of objective function values over iterations
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_2d_path(path: List[np.ndarray], objective_fn: Callable, title: str = "Optimization Path", save_path: str = None):
    """
    Visualize the optimization path in 2D space with contour lines.

    Args:
        path: List of parameter vectors representing the optimization path
        objective_fn: The objective function being minimized
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    # Extract x and y coordinates from the path
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]

    # Create a grid for contour plot
    x_min, x_max = min(min(x_coords), -2), max(max(x_coords), 2)
    y_min, y_max = min(min(y_coords), -2), max(max(y_coords), 2)

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate function values on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_fn(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)

    # Plot the optimization path
    plt.plot(x_coords, y_coords, 'ro-', markersize=5, linewidth=2, label='Optimization Path')
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, label='End')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()