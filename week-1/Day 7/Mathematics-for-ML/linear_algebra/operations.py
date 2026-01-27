"""
Additional linear algebra operations implemented from scratch.

Following the mathematical foundations of machine learning with emphasis on
educational clarity and mathematical rigor.
"""

import numpy as np
from typing import Union, List
from linear_algebra.vectors import dot_product, vector_magnitude, normalize_vector
from linear_algebra.matrices import matrix_multiply


def matrix_vector_multiply(A: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Multiply a matrix by a vector: A * v

    Args:
        A: Matrix (m x n)
        v: Vector (n x 1)

    Returns:
        Product of matrix and vector (m x 1)
    """
    A = np.asarray(A)
    v = np.asarray(v)

    if A.shape[1] != v.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {v.shape} for matrix-vector multiplication")

    # Implement matrix-vector multiplication from scratch
    m, n = A.shape

    result = np.zeros(m)
    for i in range(m):
        for j in range(n):
            result[i] += A[i, j] * v[j]

    return result


def outer_product(u: Union[np.ndarray, List], v: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute the outer product of two vectors: u ⊗ v

    Args:
        u: First vector (m x 1)
        v: Second vector (n x 1)

    Returns:
        Outer product matrix (m x n)
    """
    u = np.asarray(u)
    v = np.asarray(v)

    m = u.shape[0]
    n = v.shape[0]

    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            result[i, j] = u[i] * v[j]

    return result


def hadamard_product(A: Union[np.ndarray, List], B: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute the Hadamard (element-wise) product of two matrices: A ⊙ B

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Element-wise product of the matrices
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape != B.shape:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    m, n = A.shape
    result = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            result[i, j] = A[i, j] * B[i, j]

    return result


def frobenius_norm(A: Union[np.ndarray, List]) -> float:
    """
    Compute the Frobenius norm of a matrix: ||A||_F

    Formula: ||A||_F = sqrt(ΣΣ(A[i,j]^2))

    Args:
        A: Input matrix

    Returns:
        Frobenius norm of the matrix
    """
    A = np.asarray(A)
    m, n = A.shape

    squared_sum = 0.0
    for i in range(m):
        for j in range(n):
            squared_sum += A[i, j] ** 2

    return float(np.sqrt(squared_sum))


def matrix_rank(A: Union[np.ndarray, List]) -> int:
    """
    Compute the rank of a matrix using Gaussian elimination.

    Args:
        A: Input matrix

    Returns:
        Rank of the matrix (number of linearly independent rows/columns)
    """
    A = np.asarray(A).astype(float)
    m, n = A.shape

    # Create a copy for row reduction
    mat = A.copy()

    # Perform Gaussian elimination
    rank = min(m, n)
    for row in range(rank):
        # Check if leading element is zero
        if abs(mat[row, row]) < 1e-10:
            # Find non-zero element in the same column
            swap_row = -1
            for i in range(row + 1, m):
                if abs(mat[i, row]) > 1e-10:
                    swap_row = i
                    break

            if swap_row == -1:
                # Reduce rank and shift remaining rows
                rank -= 1
                for i in range(row, m):
                    mat[i, row] = mat[i, n - 1]
            else:
                # Swap rows
                mat[[row, swap_row]] = mat[[swap_row, row]]

        if abs(mat[row, row]) > 1e-10:
            # Eliminate column
            for i in range(m):
                if i != row and abs(mat[i, row]) > 1e-10:
                    multiplier = mat[i, row] / mat[row, row]
                    for j in range(n):
                        mat[i, j] -= multiplier * mat[row, j]

    return rank


def gram_schmidt(V: Union[np.ndarray, List]) -> np.ndarray:
    """
    Apply Gram-Schmidt process to orthogonalize a set of vectors.

    Args:
        V: Matrix where each row is a vector to orthogonalize

    Returns:
        Matrix with orthogonalized vectors
    """
    V = np.asarray(V).astype(float)
    m, n = V.shape

    U = np.zeros((m, n))

    for i in range(m):
        # Start with the current vector
        U[i] = V[i].copy()

        # Subtract projections onto previous vectors
        for j in range(i):
            # Calculate projection of V[i] onto U[j]
            proj = (dot_product(V[i], U[j]) / dot_product(U[j], U[j])) * U[j]
            U[i] -= proj

        # Normalize if not zero vector
        if vector_magnitude(U[i]) < 1e-10:
            raise ValueError("Vectors are not linearly independent")

    return U


def qr_decomposition(A: Union[np.ndarray, List]) -> tuple:
    """
    Perform QR decomposition of a matrix: A = Q * R

    Args:
        A: Input matrix

    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A = np.asarray(A).astype(float)
    m, n = A.shape

    # Use Gram-Schmidt to get Q
    Q = gram_schmidt(A.T).T  # Transpose to treat columns as vectors

    # Normalize Q columns
    for j in range(n):
        col_norm = vector_magnitude(Q[:, j])
        if col_norm > 1e-10:
            Q[:, j] /= col_norm

    # Calculate R = Q^T * A
    R = matrix_multiply(matrix_transpose(Q), A)

    # Ensure R is upper triangular (fill in lower triangle with zeros)
    for i in range(1, n):
        for j in range(i):
            R[i, j] = 0.0

    return Q, R


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