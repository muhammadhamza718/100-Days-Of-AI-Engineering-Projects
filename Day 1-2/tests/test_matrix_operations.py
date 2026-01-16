#!/usr/bin/env python3
"""
Unit Tests for Matrix Operations - NumPy Mastery Module

This test suite validates the custom matrix operations implementations
against NumPy's built-in functions to ensure mathematical correctness.
"""

import numpy as np
import pytest
from src.exercises.matrix_operations import (
    custom_dot, custom_matmul, custom_transpose
)


class TestMatrixOperations:
    """Test suite for custom matrix operations."""

    def test_custom_dot_basic(self):
        """Test basic dot product functionality."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        custom_result = custom_dot(a, b)
        expected = np.dot(a, b)

        assert np.isclose(custom_result, expected)
        assert custom_result == 32  # 1*4 + 2*5 + 3*6

    def test_custom_dot_edge_cases(self):
        """Test edge cases."""
        # Single element arrays
        assert custom_dot(np.array([5]), np.array([3])) == 15

        # Large arrays
        a = np.ones(1000)
        b = np.ones(1000)
        assert np.isclose(custom_dot(a, b), 1000.0)

    def test_custom_dot_error_handling(self):
        """Test error handling for different length arrays."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2])  # Different length

        with pytest.raises(ValueError, match="Arrays must have same length"):
            custom_dot(a, b)

    def test_custom_dot_type_validation(self):
        """Test type validation."""
        with pytest.raises(TypeError, match="Both inputs must be NumPy arrays"):
            custom_dot([1, 2, 3], np.array([4, 5, 6]))

    def test_custom_dot_1d_validation(self):
        """Test 1D array validation."""
        a_2d = np.array([[1, 2], [3, 4]])
        b_1d = np.array([1, 2])

        with pytest.raises(TypeError, match="Both inputs must be 1D arrays"):
            custom_dot(a_2d, b_1d)

    def test_custom_matmul_basic(self):
        """Test basic matrix multiplication."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        custom_result = custom_matmul(A, B)
        expected = A @ B

        assert np.allclose(custom_result, expected)

    def test_custom_matmul_different_shapes(self):
        """Test different matrix shapes."""
        # (2, 3) @ (3, 4) should work
        A = np.random.rand(2, 3)
        B = np.random.rand(3, 4)

        custom_result = custom_matmul(A, B)
        expected = A @ B

        assert custom_result.shape == (2, 4)
        assert np.allclose(custom_result, expected)

    def test_custom_matmul_identity(self):
        """Test multiplication with identity matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        I = np.eye(3)

        result = custom_matmul(A, I)
        expected = A

        assert np.allclose(result, expected)

    def test_custom_matmul_error_compatibility(self):
        """Test error for incompatible shapes."""
        A = np.array([[1, 2], [3, 4]])  # (2, 2)
        B = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2) - incompatible

        with pytest.raises(ValueError, match="Incompatible shapes"):
            custom_matmul(A, B)

    def test_custom_matmul_error_types(self):
        """Test error for wrong input types."""
        with pytest.raises(TypeError, match="Both inputs must be NumPy arrays"):
            custom_matmul([[1, 2], [3, 4]], np.array([[1, 2], [3, 4]]))

    def test_custom_matmul_error_dimensions(self):
        """Test error for non-2D arrays."""
        A_1d = np.array([1, 2, 3])
        B_2d = np.array([[1, 2], [3, 4]])

        with pytest.raises(TypeError, match="Both inputs must be 2D arrays"):
            custom_matmul(A_1d, B_2d)

    def test_custom_transpose_2d(self):
        """Test matrix transpose."""
        A = np.array([[1, 2, 3], [4, 5, 6]])

        custom_result = custom_transpose(A)
        expected = A.T

        assert custom_result.shape == (3, 2)
        assert np.allclose(custom_result, expected)

    def test_custom_transpose_identity(self):
        """Test that transpose of transpose gives original matrix."""
        A = np.random.rand(3, 4)

        double_transpose = custom_transpose(custom_transpose(A))
        original = A

        assert np.allclose(double_transpose, original)

    def test_custom_transpose_error_non_2d(self):
        """Test error for non-2D arrays."""
        A = np.array([1, 2, 3])  # 1D array

        with pytest.raises(TypeError, match="Input must be a 2D array"):
            custom_transpose(A)

    def test_custom_transpose_error_types(self):
        """Test error for wrong input types."""
        with pytest.raises(TypeError, match="Input must be a NumPy array"):
            custom_transpose([[1, 2], [3, 4]])

    def test_performance_validation(self):
        """Test that NumPy implementations are significantly faster."""
        # This is more of a demonstration than a strict test
        from timeit import timeit

        # Large arrays for meaningful comparison
        a_large = np.random.rand(1000)
        b_large = np.random.rand(1000)
        A_large = np.random.rand(50, 50)
        B_large = np.random.rand(50, 50)

        # Dot product performance - custom should be much slower
        custom_dot_time = timeit(lambda: custom_dot(a_large, b_large), number=50)
        numpy_dot_time = timeit(lambda: np.dot(a_large, b_large), number=50)

        # NumPy should be significantly faster (though we can't test exact ratio in CI)
        assert numpy_dot_time < custom_dot_time, "NumPy should be faster than custom implementation"

        # Matrix multiplication performance
        custom_matmul_time = timeit(lambda: custom_matmul(A_large, B_large), number=10)
        numpy_matmul_time = timeit(lambda: A_large @ B_large, number=10)

        # NumPy should be significantly faster
        assert numpy_matmul_time < custom_matmul_time, "NumPy should be faster than custom implementation"


def run_all_tests():
    """Run all tests and return results."""
    import unittest
    import sys

    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMatrixOperations)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Matrix Operations Unit Tests...")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\n✅ All matrix operation tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")
        sys.exit(1)