"""
Matrix operations implemented from scratch using NumPy for array operations only.

Following the mathematical foundations of machine learning with emphasis on
educational clarity and mathematical rigor.
"""

import numpy as np
from typing import Union, List
from linear_algebra.vectors import dot_product, vector_magnitude


def matrix_add(A: Union[np.ndarray, List], B: Union[np.ndarray, List]) -> np.ndarray:
    """
    Add two matrices: A + B

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Sum of the two matrices
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape != B.shape:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    return A + B


def matrix_subtract(A: Union[np.ndarray, List], B: Union[np.ndarray, List]) -> np.ndarray:
    """
    Subtract two matrices: A - B

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Difference of the two matrices
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape != B.shape:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    return A - B


def matrix_scalar_multiply(scalar: float, A: Union[np.ndarray, List]) -> np.ndarray:
    """
    Multiply a matrix by a scalar: scalar * A

    Args:
        scalar: Scalar value
        A: Matrix to multiply

    Returns:
        Scalar multiplication of the matrix
    """
    A = np.asarray(A)
    return scalar * A


def matrix_multiply(A: Union[np.ndarray, List], B: Union[np.ndarray, List]) -> np.ndarray:
    """
    Multiply two matrices: A * B

    Formula: C[i,j] = Σ(A[i,k] * B[k,j]) for k = 1 to n

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product of the two matrices (m x p)
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape} for matrix multiplication")

    # Implement matrix multiplication from scratch
    m, n = A.shape
    n2, p = B.shape

    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            # Compute C[i,j] = sum of A[i,k] * B[k,j] for k = 0 to n-1
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


def matrix_transpose(A: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute the transpose of a matrix: A^T

    Args:
        A: Input matrix (m x n)

    Returns:
        Transposed matrix (n x m)
    """
    A = np.asarray(A)
    rows, cols = A.shape
    A_T = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            A_T[j, i] = A[i, j]

    return A_T


def matrix_trace(A: Union[np.ndarray, List]) -> float:
    """
    Compute the trace of a square matrix (sum of diagonal elements).

    Args:
        A: Square input matrix

    Returns:
        Trace of the matrix (sum of diagonal elements)
    """
    A = np.asarray(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square for trace calculation, got shape {A.shape}")

    trace = 0.0
    n = A.shape[0]

    for i in range(n):
        trace += A[i, i]

    return trace


def matrix_determinant(A: Union[np.ndarray, List]) -> float:
    """
    Compute the determinant of a square matrix using cofactor expansion.

    Args:
        A: Square input matrix

    Returns:
        Determinant of the matrix
    """
    A = np.asarray(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square for determinant calculation, got shape {A.shape}")

    n = A.shape[0]

    # Base case: 1x1 matrix
    if n == 1:
        return float(A[0, 0])

    # Base case: 2x2 matrix
    if n == 2:
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    # Recursive case: nxn matrix (n > 2)
    det = 0.0
    for col in range(n):
        # Create minor matrix by removing row 0 and column 'col'
        minor = np.zeros((n-1, n-1))
        for i in range(1, n):
            for j in range(n):
                if j < col:
                    minor[i-1, j] = A[i, j]
                elif j > col:
                    minor[i-1, j-1] = A[i, j]

        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** col) * A[0, col] * matrix_determinant(minor)
        det += cofactor

    return det


def identity_matrix(size: int) -> np.ndarray:
    """
    Create an identity matrix of given size.

    Args:
        size: Size of the square identity matrix

    Returns:
        Identity matrix of specified size
    """
    I = np.zeros((size, size))
    for i in range(size):
        I[i, i] = 1.0
    return I


def matrix_inverse(A: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute the inverse of a square matrix using Gauss-Jordan elimination.

    Args:
        A: Square input matrix

    Returns:
        Inverse of the matrix
    """
    A = np.asarray(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square for inversion, got shape {A.shape}")

    n = A.shape[0]
    det = matrix_determinant(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular and cannot be inverted")

    # Create augmented matrix [A | I]
    aug = np.hstack([A, identity_matrix(n)])

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k, i]) > abs(aug[max_row, i]):
                max_row = k
        aug[[i, max_row]] = aug[[max_row, i]]

        # Scale pivot row
        pivot = aug[i, i]
        aug[i] /= pivot

        # Eliminate column
        for k in range(n):
            if k != i:
                factor = aug[k, i]
                aug[k] -= factor * aug[i]

    # Extract inverse matrix (right half of augmented matrix)
    return aug[:, n:]