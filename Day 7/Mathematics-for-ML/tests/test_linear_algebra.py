"""
Tests for linear algebra module
"""
import numpy as np
import pytest
from linear_algebra.vectors import (
    vector_add, vector_subtract, scalar_multiply, dot_product,
    vector_magnitude, normalize_vector, vector_angle, vector_projection
)
from linear_algebra.matrices import (
    matrix_add, matrix_subtract, matrix_scalar_multiply, matrix_multiply,
    matrix_transpose, matrix_trace, matrix_determinant, identity_matrix,
    matrix_inverse
)
from linear_algebra.operations import (
    matrix_vector_multiply, outer_product, hadamard_product, frobenius_norm,
    matrix_rank, gram_schmidt, qr_decomposition
)


def test_vector_add():
    """Test vector addition."""
    u = [1, 2, 3]
    v = [4, 5, 6]
    result = vector_add(u, v)
    expected = np.array([5, 7, 9])
    np.testing.assert_array_equal(result, expected)


def test_vector_subtract():
    """Test vector subtraction."""
    u = [5, 7, 9]
    v = [1, 2, 3]
    result = vector_subtract(u, v)
    expected = np.array([4, 5, 6])
    np.testing.assert_array_equal(result, expected)


def test_scalar_multiply():
    """Test scalar multiplication."""
    v = [1, 2, 3]
    result = scalar_multiply(2, v)
    expected = np.array([2, 4, 6])
    np.testing.assert_array_equal(result, expected)


def test_dot_product():
    """Test dot product."""
    u = [1, 2, 3]
    v = [4, 5, 6]
    result = dot_product(u, v)
    expected = 1*4 + 2*5 + 3*6  # 4 + 10 + 18 = 32
    assert result == expected


def test_vector_magnitude():
    """Test vector magnitude."""
    v = [3, 4]  # Should give magnitude 5 (3-4-5 triangle)
    result = vector_magnitude(v)
    expected = 5.0
    assert abs(result - expected) < 1e-10


def test_normalize_vector():
    """Test vector normalization."""
    v = [3, 4]
    result = normalize_vector(v)
    expected_magnitude = vector_magnitude(result)
    assert abs(expected_magnitude - 1.0) < 1e-10


def test_vector_angle():
    """Test vector angle calculation."""
    u = [1, 0]
    v = [0, 1]  # Should be 90 degrees or pi/2 radians
    result = vector_angle(u, v)
    expected = np.pi / 2
    assert abs(result - expected) < 1e-10


def test_vector_projection():
    """Test vector projection."""
    u = [3, 4]
    v = [1, 0]  # Project onto x-axis
    result = vector_projection(u, v)
    expected = np.array([3, 0])  # Should project to [3, 0]
    np.testing.assert_array_equal(result, expected)


def test_matrix_add():
    """Test matrix addition."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = matrix_add(A, B)
    expected = np.array([[6, 8], [10, 12]])
    np.testing.assert_array_equal(result, expected)


def test_matrix_multiply():
    """Test matrix multiplication."""
    A = [[1, 2], [3, 4]]
    B = [[2, 0], [1, 2]]
    result = matrix_multiply(A, B)
    expected = np.array([[4, 4], [10, 8]])  # [1*2+2*1, 1*0+2*2; 3*2+4*1, 3*0+4*2]
    np.testing.assert_array_equal(result, expected)


def test_matrix_transpose():
    """Test matrix transpose."""
    A = [[1, 2, 3], [4, 5, 6]]
    result = matrix_transpose(A)
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(result, expected)


def test_matrix_trace():
    """Test matrix trace."""
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = matrix_trace(A)
    expected = 1 + 5 + 9  # 15
    assert result == expected


def test_matrix_determinant():
    """Test matrix determinant."""
    A = [[1, 2], [3, 4]]
    result = matrix_determinant(A)
    expected = 1*4 - 2*3  # -2
    assert abs(result - expected) < 1e-10


def test_identity_matrix():
    """Test identity matrix creation."""
    I = identity_matrix(3)
    expected = np.eye(3)
    np.testing.assert_array_equal(I, expected)


def test_matrix_inverse():
    """Test matrix inverse."""
    A = [[4, 7], [2, 6]]
    A_inv = matrix_inverse(A)
    # Verify A * A_inv = I
    result = matrix_multiply(A, A_inv)
    expected = identity_matrix(2)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_matrix_vector_multiply():
    """Test matrix-vector multiplication."""
    A = [[1, 2], [3, 4]]
    v = [1, 2]
    result = matrix_vector_multiply(A, v)
    expected = np.array([5, 11])  # [1*1+2*2, 3*1+4*2]
    np.testing.assert_array_equal(result, expected)


def test_outer_product():
    """Test outer product."""
    u = [1, 2]
    v = [3, 4]
    result = outer_product(u, v)
    expected = np.array([[3, 4], [6, 8]])  # [1*3, 1*4; 2*3, 2*4]
    np.testing.assert_array_equal(result, expected)


def test_hadamard_product():
    """Test Hadamard (element-wise) product."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = hadamard_product(A, B)
    expected = np.array([[5, 12], [21, 32]])
    np.testing.assert_array_equal(result, expected)


def test_frobenius_norm():
    """Test Frobenius norm."""
    A = [[3, 4], [0, 0]]
    result = frobenius_norm(A)
    expected = 5.0  # sqrt(3^2 + 4^2 + 0^2 + 0^2)
    assert abs(result - expected) < 1e-10


def test_matrix_rank():
    """Test matrix rank."""
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Dependent rows
    result = matrix_rank(A)
    expected = 2  # Rank should be 2
    assert result == expected


def test_gram_schmidt():
    """Test Gram-Schmidt orthogonalization."""
    V = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    U = gram_schmidt(V)
    # Check orthogonality by verifying dot products are near zero
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            dot_prod = dot_product(U[i], U[j])
            assert abs(dot_prod) < 1e-10  # Should be nearly orthogonal


def test_qr_decomposition():
    """Test QR decomposition."""
    A = [[1, 2], [3, 4]]
    Q, R = qr_decomposition(A)

    # Verify Q is orthogonal (Q^T * Q = I)
    Q_transpose = matrix_transpose(Q)
    QQ_transpose = matrix_multiply(Q, Q_transpose)
    I = identity_matrix(len(Q))
    np.testing.assert_array_almost_equal(QQ_transpose, I, decimal=10)

    # Verify A = Q * R
    AR = matrix_multiply(Q, R)
    np.testing.assert_array_almost_equal(A, AR, decimal=10)