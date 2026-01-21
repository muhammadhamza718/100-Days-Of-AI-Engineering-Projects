"""
Vector operations implemented from scratch using NumPy for array operations only.

Following the mathematical foundations of machine learning with emphasis on
educational clarity and mathematical rigor.
"""

import numpy as np
from typing import Union, List


def vector_add(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Add two vectors: u + v

    Args:
        u: First vector
        v: Second vector

    Returns:
        Sum of the two vectors
    """
    u = np.asarray(u)
    v = np.asarray(v)

    if u.shape != v.shape:
        raise ValueError(f"Incompatible shapes: {u.shape} and {v.shape}")

    return u + v


def vector_subtract(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Subtract two vectors: u - v

    Args:
        u: First vector
        v: Second vector

    Returns:
        Difference of the two vectors
    """
    u = np.asarray(u)
    v = np.asarray(v)

    if u.shape != v.shape:
        raise ValueError(f"Incompatible shapes: {u.shape} and {v.shape}")

    return u - v


def scalar_multiply(scalar: float, v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Multiply a vector by a scalar: scalar * v

    Args:
        scalar: Scalar value
        v: Vector to multiply

    Returns:
        Scalar multiplication of the vector
    """
    v = np.asarray(v)
    return scalar * v


def dot_product(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> float:
    """
    Compute the dot product of two vectors: u · v

    Formula: u · v = Σ(u_i * v_i) for i = 1 to n

    Args:
        u: First vector
        v: Second vector

    Returns:
        Dot product (scalar value)
    """
    u = np.asarray(u)
    v = np.asarray(v)

    if u.shape != v.shape:
        raise ValueError(f"Incompatible shapes: {u.shape} and {v.shape}")

    return float(np.sum(u * v))


def vector_magnitude(v: Union[np.ndarray, List]) -> float:
    """
    Compute the magnitude (norm) of a vector: ||v||

    Formula: ||v|| = sqrt(Σ(v_i^2)) for i = 1 to n

    Args:
        v: Input vector

    Returns:
        Magnitude of the vector
    """
    v = np.asarray(v)
    return float(np.sqrt(np.sum(v ** 2)))


def normalize_vector(v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Normalize a vector to unit length: v / ||v||

    Args:
        v: Input vector

    Returns:
        Normalized vector with unit magnitude
    """
    v = np.asarray(v)
    mag = vector_magnitude(v)

    if mag == 0:
        raise ValueError("Cannot normalize zero vector")

    return v / mag


def vector_angle(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> float:
    """
    Compute the angle between two vectors in radians.

    Formula: θ = arccos((u · v) / (||u|| * ||v||))

    Args:
        u: First vector
        v: Second vector

    Returns:
        Angle between vectors in radians
    """
    u = np.asarray(u)
    v = np.asarray(v)

    dot_prod = dot_product(u, v)
    mag_u = vector_magnitude(u)
    mag_v = vector_magnitude(v)

    if mag_u == 0 or mag_v == 0:
        raise ValueError("Cannot compute angle with zero vector")

    cos_theta = dot_prod / (mag_u * mag_v)
    # Clamp to [-1, 1] to handle floating point errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return float(np.arccos(cos_theta))


def vector_projection(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Project vector u onto vector v.

    Formula: proj_v(u) = ((u · v) / ||v||²) * v

    Args:
        u: Vector to project
        v: Vector to project onto

    Returns:
        Projection of u onto v
    """
    u = np.asarray(u)
    v = np.asarray(v)

    v_mag_sq = vector_magnitude(v) ** 2

    if v_mag_sq == 0:
        raise ValueError("Cannot project onto zero vector")

    dot_prod = dot_product(u, v)
    scalar_proj = dot_prod / v_mag_sq

    return scalar_proj * v