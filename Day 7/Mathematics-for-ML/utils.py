"""
Common utilities for mathematical operations
"""
import numpy as np


def normalize_vector(v):
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def safe_divide(a, b, epsilon=1e-8):
    """
    Safely divide two numbers, avoiding division by zero.

    Args:
        a: Numerator
        b: Denominator
        epsilon: Small value to avoid division by zero

    Returns:
        Result of division
    """
    return a / (b + epsilon)


def clamp(value, min_val, max_val):
    """
    Clamp a value between min and max.

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def is_symmetric(matrix, rtol=1e-5, atol=1e-8):
    """
    Check if a matrix is symmetric.

    Args:
        matrix: Input matrix
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if matrix is symmetric
    """
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)