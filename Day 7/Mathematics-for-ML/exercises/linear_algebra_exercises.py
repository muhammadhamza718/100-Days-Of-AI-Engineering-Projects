"""
Exercise: Linear Algebra Operations

This exercise demonstrates fundamental linear algebra operations.
"""

import numpy as np
from linear_algebra.vectors import (
    vector_add, vector_subtract, scalar_multiply, dot_product,
    vector_magnitude, normalize_vector
)
from linear_algebra.matrices import (
    matrix_add, matrix_multiply, matrix_transpose, matrix_trace
)
from linear_algebra.operations import (
    matrix_vector_multiply, outer_product, frobenius_norm
)


def main():
    print("=== Linear Algebra Operations Exercise ===\n")

    # Vector operations
    print("1. Vector Operations:")
    u = [1, 2, 3]
    v = [4, 5, 6]
    print(f"   u = {u}")
    print(f"   v = {v}")

    result = vector_add(u, v)
    print(f"   u + v = {result}")

    result = dot_product(u, v)
    print(f"   u · v = {result}")

    result = vector_magnitude(u)
    print(f"   ||u|| = {result}")

    result = normalize_vector(u)
    print(f"   normalized u = {result}")
    print(f"   magnitude of normalized u = {vector_magnitude(result)}")
    print()

    # Matrix operations
    print("2. Matrix Operations:")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print(f"   A = {A}")
    print(f"   B = {B}")

    result = matrix_add(A, B)
    print(f"   A + B = {result}")

    result = matrix_multiply(A, B)
    print(f"   A * B = {result}")

    result = matrix_transpose(A)
    print(f"   A^T = {result}")

    result = matrix_trace(A)
    print(f"   tr(A) = {result}")
    print()

    # Matrix-vector operations
    print("3. Matrix-Vector Operations:")
    v_vec = [1, 2]
    print(f"   A = {A}")
    print(f"   v = {v_vec}")

    result = matrix_vector_multiply(A, v_vec)
    print(f"   A * v = {result}")
    print()

    # Special operations
    print("4. Special Operations:")
    print(f"   u = {u}")
    print(f"   v = {v[:2]} (first 2 elements)")

    result = outer_product(u[:2], v[:2])  # Same dimension for outer product
    print(f"   u (outer product) v = {result}")

    result = frobenius_norm(A)
    print(f"   ||A||_F = {result}")
    print()

    print("Linear algebra operations completed successfully!")


if __name__ == "__main__":
    main()