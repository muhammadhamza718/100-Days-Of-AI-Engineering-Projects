"""
Matrix operations utilities for regression implementation.

This module provides fundamental matrix operations implemented from scratch
to support the regression algorithms without relying on high-level libraries
for core mathematical computations.
"""

import numpy as np
from typing import Tuple


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication of two matrices A and B.

    Args:
        A (np.ndarray): Left matrix of shape (m, n)
        B (np.ndarray): Right matrix of shape (n, p)

    Returns:
        np.ndarray: Result of A * B with shape (m, p)
    """
    # Manual implementation of matrix multiplication
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError(f"Incompatible dimensions for matrix multiplication: {A.shape} and {B.shape}")

    # Initialize result matrix
    result = np.zeros((rows_A, cols_B))

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    return result


def transpose(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the transpose of a matrix.

    Args:
        matrix (np.ndarray): Input matrix of shape (m, n)

    Returns:
        np.ndarray: Transposed matrix of shape (n, m)
    """
    rows, cols = matrix.shape
    transposed = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = matrix[i, j]

    return transposed


def vector_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        a (np.ndarray): First vector of shape (n,)
        b (np.ndarray): Second vector of shape (n,)

    Returns:
        float: Dot product of the two vectors
    """
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same shape for dot product: {a.shape} vs {b.shape}")

    if len(a.shape) != 1:
        raise ValueError(f"Dot product expects 1D vectors, got shapes: {a.shape}")

    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def element_wise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perform element-wise multiplication of two arrays.

    Args:
        a (np.ndarray): First array
        b (np.ndarray): Second array

    Returns:
        np.ndarray: Element-wise product of the arrays
    """
    if a.shape != b.shape:
        raise ValueError(f"Arrays must have same shape for element-wise multiplication: {a.shape} vs {b.shape}")

    # Use numpy for efficiency while maintaining our own implementation structure
    return a * b


def sum_array(arr: np.ndarray) -> float:
    """
    Compute the sum of all elements in an array.

    Args:
        arr (np.ndarray): Input array

    Returns:
        float: Sum of all elements
    """
    total = 0.0
    for element in arr.flat:
        total += element
    return total


def mean_axis(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the mean of an array along a specified axis.

    Args:
        arr (np.ndarray): Input array
        axis (int): Axis along which to compute the mean

    Returns:
        np.ndarray: Array of means
    """
    if axis == 0:
        # Compute mean along columns (for each feature)
        means = np.zeros(arr.shape[1])
        for j in range(arr.shape[1]):
            col_sum = 0.0
            for i in range(arr.shape[0]):
                col_sum += arr[i, j]
            means[j] = col_sum / arr.shape[0]
        return means
    elif axis == 1:
        # Compute mean along rows
        means = np.zeros(arr.shape[0])
        for i in range(arr.shape[0]):
            row_sum = 0.0
            for j in range(arr.shape[1]):
                row_sum += arr[i, j]
            means[i] = row_sum / arr.shape[1]
        return means
    else:
        raise ValueError(f"Unsupported axis: {axis}. Only 0 and 1 supported.")


def std_axis(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the standard deviation of an array along a specified axis.

    Args:
        arr (np.ndarray): Input array
        axis (int): Axis along which to compute the standard deviation

    Returns:
        np.ndarray: Array of standard deviations
    """
    means = mean_axis(arr, axis)

    if axis == 0:
        # Compute std along columns (for each feature)
        stds = np.zeros(arr.shape[1])
        for j in range(arr.shape[1]):
            col_sq_diff_sum = 0.0
            for i in range(arr.shape[0]):
                diff = arr[i, j] - means[j]
                col_sq_diff_sum += diff * diff
            stds[j] = np.sqrt(col_sq_diff_sum / arr.shape[0])
        return stds
    elif axis == 1:
        # Compute std along rows
        stds = np.zeros(arr.shape[0])
        for i in range(arr.shape[0]):
            row_sq_diff_sum = 0.0
            for j in range(arr.shape[1]):
                diff = arr[i, j] - means[i]
                row_sq_diff_sum += diff * diff
            stds[i] = np.sqrt(row_sq_diff_sum / arr.shape[1])
        return stds
    else:
        raise ValueError(f"Unsupported axis: {axis}. Only 0 and 1 supported.")