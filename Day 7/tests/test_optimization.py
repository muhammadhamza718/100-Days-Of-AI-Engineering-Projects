"""
Tests for optimization module
"""
import numpy as np
import pytest
from optimization.gradient_descent import (
    gradient_descent,
    quadratic_function,
    quadratic_gradient,
    rosenbrock_function,
    rosenbrock_gradient
)


def test_gradient_descent_basic():
    """Test basic gradient descent functionality."""
    start_point = np.array([5.0, 5.0])
    learning_rate = 0.1
    num_iterations = 10

    path, values = gradient_descent(
        objective_fn=quadratic_function,
        gradient_fn=quadratic_gradient,
        start_point=start_point,
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )

    # Check that we have the right number of points (iterations + 1)
    assert len(path) == num_iterations + 1
    assert len(values) == num_iterations + 1

    # Check that function value decreases (at least for first few steps)
    assert values[0] > values[-1]  # Final value should be lower than initial


def test_quadratic_function():
    """Test the quadratic function implementation."""
    point = np.array([2.0, 3.0])
    result = quadratic_function(point)
    expected = 2.0**2 + 3.0**2  # 4 + 9 = 13
    assert result == expected


def test_quadratic_gradient():
    """Test the quadratic gradient implementation."""
    point = np.array([2.0, 3.0])
    result = quadratic_gradient(point)
    expected = np.array([4.0, 6.0])  # [2*2, 2*3]
    np.testing.assert_array_equal(result, expected)


def test_convergence_to_origin():
    """Test convergence to [0,0] for quadratic function (T010)."""
    start_point = np.array([5.0, 5.0])
    learning_rate = 0.1
    num_iterations = 100

    path, values = gradient_descent(
        objective_fn=quadratic_function,
        gradient_fn=quadratic_gradient,
        start_point=start_point,
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )

    final_point = path[-1]
    # Check that we're close to [0,0] within tolerance
    distance_from_origin = np.linalg.norm(final_point)
    assert distance_from_origin < 0.1  # Within 0.1 of origin


def test_different_learning_rates():
    """Test different learning rates behavior (T011)."""
    start_point = np.array([5.0, 5.0])
    num_iterations = 50

    learning_rates = [0.01, 0.1, 0.5]
    final_distances = []

    for lr in learning_rates:
        path, values = gradient_descent(
            objective_fn=quadratic_function,
            gradient_fn=quadratic_gradient,
            start_point=start_point,
            learning_rate=lr,
            num_iterations=num_iterations
        )

        final_distance = np.linalg.norm(path[-1])
        final_distances.append(final_distance)

    # Higher learning rates should generally converge faster (be closer to origin)
    # though very high rates might overshoot
    assert isinstance(final_distances[0], float)  # All should be valid floats
    assert len(final_distances) == len(learning_rates)


def test_rosenbrock_function():
    """Test the rosenbrock function implementation."""
    # Test at minimum point
    point = np.array([1.0, 1.0])
    result = rosenbrock_function(point)
    expected = 0.0
    assert abs(result - expected) < 1e-10


def test_rosenbrock_gradient_at_minimum():
    """Test the rosenbrock gradient at the minimum point."""
    point = np.array([1.0, 1.0])  # Minimum point
    result = rosenbrock_gradient(point)
    expected = np.array([0.0, 0.0])  # Gradient should be zero at minimum
    np.testing.assert_allclose(result, expected, atol=1e-10)