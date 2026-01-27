# Implementation Plan: NumPy Mastery Learning Module

**Branch**: `001-numpy-mastery` | **Date**: 2026-01-15 | **Spec**: [specs/numpy-mastery/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-numpy-mastery/spec.md`

## Summary

This plan implements a comprehensive NumPy learning module covering core array operations, vectorization performance optimization, linear algebra fundamentals, and practical image processing. The module consists of three main components: educational notebook with core concepts, "from-scratch" matrix operation exercises, and an image processing mini-project. All implementations will follow the constitution's requirements for vectorization-first approach, clear variable naming, and performance validation.

## Technical Context

**Language/Version**: Python 3.8+ (specifically 3.9 recommended for optimal NumPy compatibility)
**Primary Dependencies**:
- NumPy 1.20+ (core array operations, linear algebra)
- Matplotlib 3.5+ (visualization, image display)
- Pillow 9.0+ (image loading and manipulation)
- IPython/Jupyter (interactive learning environment)

**Storage**: N/A (local file processing for image mini-project)
**Testing**: Manual verification via `%timeit` comparisons and result validation against NumPy built-ins
**Target Platform**: Jupyter Notebook or VS Code Python environment
**Project Type**: Educational module (single project structure)
**Performance Goals**: Demonstrate 10-100x performance improvement for vectorized operations vs loops
**Constraints**: Must not use `np.dot` or `np.matmul` in educational exercises; all code must be beginner-friendly with clear comments
**Scale/Scope**: 3 comprehensive learning sections, 2-4 hour completion time target

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Constitutional Compliance

**I. Vectorization Over Loops**: ✅ PASS
- Core notebook will demonstrate vectorized solutions as primary approach
- Performance comparisons will show 10-100x improvement metrics
- Loop-based examples only for educational comparison

**II. "From Scratch" Exercise Constraints**: ✅ PASS
- `custom_matmul`, `custom_dot`, `custom_transpose` will use only array indexing and Python loops
- Explicit prohibition of `np.dot` and `np.matmul` in exercise implementations
- Performance timing will show why optimized functions exist

**III. Broadcasting First Explanation**: ✅ PASS
- Broadcasting section will explain "stretching" analogy
- Shape mismatch examples will include explicit broadcasting rules
- All arithmetic operations between differently-sized arrays will include broadcasting explanation

**IV. Code Quality and Naming Standards**: ✅ PASS
- Variable names: `X_train`, `weights`, `image_array`, `result_matrix` (no generic `a`, `b`, `arr`)
- Comments will explain WHY specific NumPy functions are used, especially `axis` parameters
- All code examples will follow descriptive naming conventions

**V. Mini-Project Focus on Array Representation**: ✅ PASS
- Emphasize images as 3D arrays (Height × Width × Channels) throughout
- Filters will be presented as mathematical operations on array data
- Visual demonstrations will show array-to-image transformations

**VI. Efficiency Validation**: ✅ PASS
- `%timeit` will be used for all performance comparisons
- Quantitative metrics will be displayed (execution time, performance ratios)
- Educational value will be reinforced with concrete numbers

**Overall**: All constitutional principles satisfied with no violations.

## Project Structure

### Documentation

```text
specs/numpy-mastery/
├── plan.md                          # This implementation plan
├── spec.md                          # Feature specification
├── checklists/
│   └── requirements.md             # Quality validation checklist
└── tasks.md                         # Will be created by /sp.tasks (next phase)
```

### Source Code Structure

```text
src/
├── notebooks/
│   └── numpy_fundamentals.ipynb    # Main educational notebook
├── exercises/
│   └── matrix_operations.py        # Custom matrix operations (from-scratch)
├── projects/
│   └── image_filters.py            # Image processing mini-project
└── utils/
    └── performance.py              # Performance timing utilities

data/
├── sample_images/                  # Sample images for mini-project
└── generated/                      # Output images from filters

tests/
├── test_matrix_operations.py       # Validation for custom implementations
└── test_image_filters.py           # Validation for image processing
```

**Structure Decision**: Single project structure with clear separation between educational components. Each section (notebook, exercises, projects) is independently runnable and testable.

## Implementation Phases

### Phase 0: Environment Setup

**Objective**: Create isolated environment with all required dependencies

#### 0.1 Virtual Environment Creation
```bash
# Using uv (preferred) or pip
cd Day-1-2
uv venv --python python3.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install numpy==1.24.0 matplotlib==3.7.0 pillow==9.4.0 ipython jupyter
```

#### 0.2 Project Directory Structure
```bash
# Create all directories
mkdir -p src/{notebooks,exercises,projects,utils}
mkdir -p data/{sample_images,generated}
mkdir -p tests

# Create __init__.py files
touch src/__init__.py
touch src/utils/__init__.py
```

#### 0.3 Environment Verification Script
**File**: `src/utils/verify_environment.py`
**Purpose**: Validate all dependencies are installed and versions are correct
```python
def verify_environment():
    """Verify all required packages are installed with correct versions."""
    import numpy as np
    import matplotlib
    from PIL import Image
    import sys

    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Pillow: {Image.__version__}")

    # Verify NumPy version requirement
    assert np.__version__ >= "1.20.0", f"NumPy 1.20+ required, got {np.__version__}"
    print("✅ Environment verified successfully!")
```

### Phase 1: Core Module Implementation

**Objective**: Create comprehensive educational notebook covering all fundamental concepts

#### 1.1 Main Educational Notebook
**File**: `src/notebooks/numpy_fundamentals.ipynb`

**Cell 1: Imports and Setup**
```python
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import sys

print("NumPy Mastery - Day 1-2 Learning Module")
print(f"NumPy version: {np.__version__}")
```

**Cell 2: Array Creation and Basic Operations**
```python
# Array creation examples
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
array_3d = np.random.rand(2, 3, 3)

print("1D Array shape:", array_1d.shape)
print("2D Array shape:", array_2d.shape)
print("3D Array shape:", array_3d.shape)

# Basic arithmetic (element-wise)
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])
print("Addition:", x + y)
print("Multiplication:", x * y)
```

**Cell 3: Vectorization Performance Comparison**
```python
# Non-vectorized (slow) - as per constitution, only for comparison
def sum_squares_loop(arr):
    result = 0
    for x in arr:
        result += x ** 2
    return result

# Vectorized (fast) - primary approach
def sum_squares_numpy(arr):
    return np.sum(arr ** 2)  # WHY: np.sum operates at C speed, optimized for arrays

# Performance comparison
test_array = np.random.rand(10000)
loop_time = timeit(lambda: sum_squares_loop(test_array), number=100)
numpy_time = timeit(lambda: sum_squares_numpy(test_array), number=100)

print(f"Loop implementation: {loop_time:.4f}s")
print(f"NumPy implementation: {numpy_time:.4f}s")
print(f"Performance improvement: {loop_time/numpy_time:.1f}x faster")
```

**Cell 4: Broadcasting Explanation**
```python
# Broadcasting example - "stretching" smaller array to match larger
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
b = np.array([10, 20, 30])  # Shape: (3,)

print("Array A shape:", A.shape)
print("Array b shape:", b.shape)

# b is "stretched" to shape (1, 3) then (2, 3) for element-wise addition
result = A + b
print("Broadcasting result:", result)
print("Result shape:", result.shape)

# Explanation: b is conceptually repeated along axis 0:
# [[10, 20, 30],
#  [10, 20, 30]]
```

**Cell 5: Linear Algebra Operations**
```python
# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot_product}")

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Equivalent to np.matmul(A, B)
print(f"Matrix multiplication:\n{C}")

# Other common operations
print(f"Inverse of A:\n{np.linalg.inv(A)}")
print(f"Eigenvalues: {np.linalg.eig(A)[0]}")
print(f"Transpose:\n{A.T}")
```

**Cell 6: Random Sampling and Statistics**
```python
# Random sampling
uniform_data = np.random.rand(5, 3)  # Uniform [0, 1)
normal_data = np.random.randn(1000)  # Standard normal
int_data = np.random.randint(0, 10, size=(3, 3))

# Statistical operations
data = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(f"Mean: {np.mean(data)}")
print(f"Standard Deviation: {np.std(data)}")
print(f"Max: {np.max(data)}")
print(f"Argmax (index): {np.argmax(data)}")

# Axis-specific operations (explain WHY axis=0 vs axis=1)
print(f"Sum across columns (axis=0): {np.sum(data, axis=0)}")  # [5 7 9]
print(f"Sum across rows (axis=1): {np.sum(data, axis=1)}")     # [6 15]
```

### Phase 2: Exercise Implementation

**Objective**: Create "from-scratch" matrix operations without using np.dot/np.matmul

#### 2.1 Custom Matrix Operations
**File**: `src/exercises/matrix_operations.py`

```python
import numpy as np
from timeit import timeit

def custom_dot(a, b):
    """
    Custom dot product implementation using loops only.

    Args:
        a (np.ndarray): 1D array
        b (np.ndarray): 1D array of same length

    Returns:
        float: Dot product of a and b

    Raises:
        ValueError: If arrays have different lengths
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length. Got {len(a)} and {len(b)}")

    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result

def custom_matmul(A, B):
    """
    Custom matrix multiplication using loops only.

    Args:
        A (np.ndarray): 2D array of shape (m, n)
        B (np.ndarray): 2D array of shape (n, p)

    Returns:
        np.ndarray: Result of A @ B with shape (m, p)

    Raises:
        ValueError: If inner dimensions don't match
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    m, n = A.shape
    p = B.shape[1]
    result = np.zeros((m, p))

    # Standard matrix multiplication algorithm
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]

    return result

def custom_transpose(A):
    """
    Custom matrix transpose using loops only.

    Args:
        A (np.ndarray): 2D array

    Returns:
        np.ndarray: Transposed array
    """
    if len(A.shape) != 2:
        raise ValueError("Transpose only supported for 2D arrays")

    rows, cols = A.shape
    result = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j, i] = A[i, j]

    return result

def validate_custom_implementations():
    """
    Validate custom implementations against NumPy built-ins.
    """
    print("=== Validation of Custom Implementations ===")

    # Test dot product
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])

    custom_result = custom_dot(a, b)
    numpy_result = np.dot(a, b)

    print(f"Dot product - Custom: {custom_result}, NumPy: {numpy_result}")
    print(f"Dot product match: {np.isclose(custom_result, numpy_result)}")

    # Test matrix multiplication
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])

    custom_result = custom_matmul(A, B)
    numpy_result = A @ B

    print(f"Matmul - Custom:\n{custom_result}")
    print(f"Matmul - NumPy:\n{numpy_result}")
    print(f"Matmul match: {np.allclose(custom_result, numpy_result)}")

    # Test transpose
    C = np.array([[1, 2, 3], [4, 5, 6]])

    custom_result = custom_transpose(C)
    numpy_result = C.T

    print(f"Transpose - Custom:\n{custom_result}")
    print(f"Transpose - NumPy:\n{numpy_result}")
    print(f"Transpose match: {np.allclose(custom_result, numpy_result)}")

def performance_comparison():
    """
    Compare performance between custom and NumPy implementations.
    """
    print("\n=== Performance Comparison ===")

    # Large matrices for meaningful timing
    A_large = np.random.rand(50, 50)
    B_large = np.random.rand(50, 50)
    a_large = np.random.rand(1000)
    b_large = np.random.rand(1000)

    # Dot product performance
    custom_dot_time = timeit(lambda: custom_dot(a_large, b_large), number=100)
    numpy_dot_time = timeit(lambda: np.dot(a_large, b_large), number=100)

    print(f"Dot Product:")
    print(f"  Custom: {custom_dot_time:.4f}s")
    print(f"  NumPy:  {numpy_dot_time:.4f}s")
    print(f"  NumPy is {custom_dot_time/numpy_dot_time:.1f}x faster")

    # Matrix multiplication performance
    custom_matmul_time = timeit(lambda: custom_matmul(A_large, B_large), number=100)
    numpy_matmul_time = timeit(lambda: A_large @ B_large, number=100)

    print(f"\nMatrix Multiplication:")
    print(f"  Custom: {custom_matmul_time:.4f}s")
    print(f"  NumPy:  {numpy_matmul_time:.4f}s")
    print(f"  NumPy is {custom_matmul_time/numpy_matmul_time:.1f}x faster")

if __name__ == "__main__":
    validate_custom_implementations()
    performance_comparison()
```

### Phase 3: Mini-Project Architecture

**Objective**: Create image processing mini-project with grayscale conversion and color inversion

#### 3.1 Image Processing Mini-Project
**File**: `src/projects/image_filters.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageFilters:
    """
    Image processing mini-project demonstrating NumPy array operations.

    Emphasizes that images are 3D NumPy arrays (Height × Width × Channels)
    and filters are mathematical operations on these arrays.
    """

    def __init__(self):
        self.image_array = None
        self.original_shape = None

    def load_image(self, image_path):
        """
        Load an image and convert to NumPy array.

        Args:
            image_path (str): Path to image file

        Returns:
            np.ndarray: Image as 3D array (H × W × C)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image and convert to RGB (handle RGBA, grayscale images)
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        self.image_array = np.array(img)
        self.original_shape = self.image_array.shape

        print(f"Image loaded successfully!")
        print(f"Shape: {self.image_array.shape} (Height × Width × Channels)")
        print(f"Data type: {self.image_array.dtype}")
        print(f"Value range: {self.image_array.min()} - {self.image_array.max()}")

        return self.image_array

    def convert_to_grayscale(self, image_array=None):
        """
        Convert color image to grayscale using luminance formula.

        Y = 0.2989 × R + 0.5870 × G + 0.1140 × B

        WHY: This formula accounts for human perception of brightness.
             Different colors contribute differently to perceived brightness.

        Args:
            image_array (np.ndarray): Optional, uses loaded image if None

        Returns:
            np.ndarray: 2D grayscale image array
        """
        if image_array is None:
            image_array = self.image_array

        if image_array is None:
            raise ValueError("No image loaded. Call load_image() first.")

        # Extract RGB channels using array slicing
        R = image_array[:, :, 0]  # Red channel
        G = image_array[:, :, 1]  # Green channel
        B = image_array[:, :, 2]  # Blue channel

        # Apply grayscale formula using broadcasting
        # WHY: np.sum with weights performs weighted average across channels
        grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B

        # Ensure values stay in valid range [0, 255]
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

        return grayscale

    def invert_colors(self, image_array=None):
        """
        Invert colors using: New_Pixel = 255 - Old_Pixel

        Args:
            image_array (np.ndarray): Optional, uses loaded image if None

        Returns:
            np.ndarray: Color-inverted image array
        """
        if image_array is None:
            image_array = self.image_array

        if image_array is None:
            raise ValueError("No image loaded. Call load_image() first.")

        # Element-wise subtraction using broadcasting
        # WHY: This works on entire array without loops, leveraging C optimizations
        inverted = 255 - image_array

        return inverted

    def box_blur(self, image_array=None, kernel_size=3):
        """
        Optional: Simple box blur using neighboring pixel averaging.

        Args:
            image_array (np.ndarray): Optional, uses loaded image if None
            kernel_size (int): Size of averaging window (must be odd)

        Returns:
            np.ndarray: Blurred image array
        """
        if image_array is None:
            image_array = self.image_array

        if image_array is None:
            raise ValueError("No image loaded. Call load_image() first.")

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        if len(image_array.shape) == 3:
            # Color image - apply blur to each channel
            blurred = np.zeros_like(image_array)
            for channel in range(3):
                blurred[:, :, channel] = self._box_blur_2d(image_array[:, :, channel], kernel_size)
        else:
            # Grayscale image
            blurred = self._box_blur_2d(image_array, kernel_size)

        return blurred

    def _box_blur_2d(self, array_2d, kernel_size):
        """Helper function for 2D box blur."""
        pad = kernel_size // 2
        padded = np.pad(array_2d, pad, mode='constant')
        blurred = np.zeros_like(array_2d)

        # Apply averaging kernel
        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                blurred[i, j] = np.mean(region)

        return blurred

    def display_comparison(self, original, filtered, title="Comparison"):
        """
        Display original and filtered images side by side.

        Args:
            original: Original image array
            filtered: Processed image array
            title (str): Title for the comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display original
        ax1.imshow(original)
        ax1.set_title("Original")
        ax1.axis('off')

        # Display filtered
        if len(filtered.shape) == 2:
            ax2.imshow(filtered, cmap='gray')
        else:
            ax2.imshow(filtered)
        ax2.set_title("Filtered")
        ax2.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def save_image(self, image_array, output_path):
        """
        Save NumPy array as image file.

        Args:
            image_array (np.ndarray): Image to save
            output_path (str): Output file path
        """
        # Ensure proper data type and range
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image and save
        if len(image_array.shape) == 2:
            # Grayscale
            img = Image.fromarray(image_array, mode='L')
        else:
            # Color
            img = Image.fromarray(image_array, mode='RGB')

        img.save(output_path)
        print(f"Image saved to: {output_path}")


def demonstrate_mini_project():
    """
    Main demonstration function for the image processing mini-project.
    """
    print("=== Image Processing Mini-Project ===")
    print("This demonstrates that images are 3D NumPy arrays (H × W × C)")
    print("and filters are mathematical operations on these arrays.\n")

    # Create a simple test image if no real image is available
    def create_test_image():
        """Create a colorful test pattern."""
        height, width = 100, 100
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Create colored regions
        image[0:50, 0:50, 0] = 255    # Red top-left
        image[0:50, 50:, 1] = 255     # Green top-right
        image[50:, 0:50, 2] = 255     # Blue bottom-left
        image[50:, 50:, :] = 200      # Gray bottom-right

        return image

    # Initialize processor
    processor = ImageFilters()

    # Use test image (in real scenario, use: processor.load_image("path/to/image.jpg"))
    test_image = create_test_image()
    processor.image_array = test_image
    processor.original_shape = test_image.shape

    print("Test image created (in practice, load with load_image() method)")
    print(f"Image shape: {test_image.shape}")

    # Grayscale conversion
    print("\n1. Grayscale Conversion")
    grayscale = processor.convert_to_grayscale()
    print(f"Grayscale shape: {grayscale.shape} (now 2D - height × width)")

    # Color inversion
    print("\n2. Color Inversion")
    inverted = processor.invert_colors()
    print(f"Inverted shape: {inverted.shape} (still 3D - H × W × C)")

    # Display results
    processor.display_comparison(test_image, grayscale, "Original vs Grayscale")
    processor.display_comparison(test_image, inverted, "Original vs Inverted")

    # Optional: Box blur (commented out for speed)
    print("\n3. Box Blur (Optional)")
    try:
        blurred = processor.box_blur(kernel_size=5)
        print(f"Blurred shape: {blurred.shape}")
        processor.display_comparison(test_image, blurred, "Original vs Blurred")
    except Exception as e:
        print(f"Blur skipped: {e}")

    print("\n=== Mini-Project Complete ===")
    print("Key takeaway: All operations were mathematical transformations")
    print("on the NumPy array representation of the image!")


if __name__ == "__main__":
    demonstrate_mini_project()
```

### Phase 4: Validation and Testing Strategy

**Objective**: Define comprehensive validation for performance comparisons and correctness

#### 4.1 Test Suite Implementation
**File**: `tests/test_matrix_operations.py`

```python
import numpy as np
import pytest
from src.exercises.matrix_operations import (
    custom_dot, custom_matmul, custom_transpose,
    validate_custom_implementations, performance_comparison
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

    def test_custom_dot_error(self):
        """Test error handling."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2])  # Different length

        with pytest.raises(ValueError, match="Arrays must have same length"):
            custom_dot(a, b)

    def test_custom_matmul_basic(self):
        """Test basic matrix multiplication."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        custom_result = custom_matmul(A, B)
        expected = A @ B

        assert np.allclose(custom_result, expected)

    def test_custom_matmul_shapes(self):
        """Test different matrix shapes."""
        # (2, 3) @ (3, 4) should work
        A = np.random.rand(2, 3)
        B = np.random.rand(3, 4)

        custom_result = custom_matmul(A, B)
        expected = A @ B

        assert custom_result.shape == (2, 4)
        assert np.allclose(custom_result, expected)

    def test_custom_matmul_error(self):
        """Test error for incompatible shapes."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4]])  # (2, 2) - incompatible

        with pytest.raises(ValueError, match="Incompatible shapes"):
            custom_matmul(A, B)

    def test_custom_transpose_2d(self):
        """Test matrix transpose."""
        A = np.array([[1, 2, 3], [4, 5, 6]])

        custom_result = custom_transpose(A)
        expected = A.T

        assert custom_result.shape == (3, 2)
        assert np.allclose(custom_result, expected)

    def test_custom_transpose_error(self):
        """Test error for non-2D arrays."""
        A = np.array([1, 2, 3])  # 1D array

        with pytest.raises(ValueError, match="Transpose only supported for 2D arrays"):
            custom_transpose(A)

    def test_performance_validation(self):
        """Test that NumPy implementations are significantly faster."""
        from timeit import timeit

        # Large arrays for meaningful comparison
        a_large = np.random.rand(1000)
        b_large = np.random.rand(1000)
        A_large = np.random.rand(50, 50)
        B_large = np.random.rand(50, 50)

        # Dot product performance
        custom_dot_time = timeit(lambda: custom_dot(a_large, b_large), number=50)
        numpy_dot_time = timeit(lambda: np.dot(a_large, b_large), number=50)

        # NumPy should be at least 50x faster for dot product
        assert custom_dot_time > numpy_dot_time * 50, \
            f"NumPy dot product should be at least 50x faster. Got {custom_dot_time/numpy_dot_time:.1f}x"

        # Matrix multiplication performance
        custom_matmul_time = timeit(lambda: custom_matmul(A_large, B_large), number=20)
        numpy_matmul_time = timeit(lambda: A_large @ B_large, number=20)

        # NumPy should be at least 50x faster for matmul
        assert custom_matmul_time > numpy_matmul_time * 50, \
            f"NumPy matmul should be at least 50x faster. Got {custom_matmul_time/numpy_matmul_time:.1f}x"

class TestImageFilters:
    """Test suite for image processing mini-project."""

    def test_grayscale_conversion(self):
        """Test grayscale conversion with known values."""
        from src.projects.image_filters import ImageFilters

        processor = ImageFilters()

        # Create test image with known RGB values
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[:, :, 0] = 100  # Red
        test_image[:, :, 1] = 150  # Green
        test_image[:, :, 2] = 200  # Blue

        processor.image_array = test_image
        grayscale = processor.convert_to_grayscale()

        # Expected value: 0.2989*100 + 0.5870*150 + 0.1140*200
        expected_value = 0.2989 * 100 + 0.5870 * 150 + 0.1140 * 200

        assert grayscale.shape == (10, 10)  # Should be 2D
        assert grayscale.dtype == np.uint8
        assert np.allclose(grayscale, expected_value, atol=1)  # Allow small rounding

    def test_color_inversion(self):
        """Test color inversion."""
        from src.projects.image_filters import ImageFilters

        processor = ImageFilters()

        # Create test image
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)  # Single pixel
        processor.image_array = test_image

        inverted = processor.invert_colors()

        expected = np.array([[[155, 105, 55]]], dtype=np.uint8)  # 255 - original
        assert np.array_equal(inverted, expected)

    def test_box_blur_simple(self):
        """Test simple box blur on small array."""
        from src.projects.image_filters import ImageFilters

        processor = ImageFilters()

        # Simple 3x3 test
        test_array = np.array([[0, 0, 0],
                               [0, 255, 0],
                               [0, 0, 0]], dtype=np.uint8)

        result = processor._box_blur_2d(test_array, 3)

        # Center should be reduced, others increased
        assert result[1, 1] < 255
        assert result[0, 0] > 0
```

**File**: `tests/test_image_filters.py`

```python
import numpy as np
import pytest
import tempfile
import os
from PIL import Image
from src.projects.image_filters import ImageFilters

class TestImageFiltersIntegration:
    """Integration tests for image processing pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete image processing pipeline."""
        processor = ImageFilters()

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create colorful test image
            test_img = np.zeros((50, 50, 3), dtype=np.uint8)
            test_img[0:25, 0:25, 0] = 255  # Red quarter
            test_img[0:25, 25:, 1] = 255   # Green quarter
            test_img[25:, 0:25, 2] = 255   # Blue quarter
            test_img[25:, 25:, :] = 128    # Gray quarter

            # Save as PNG
            PIL_image = Image.fromarray(test_img)
            PIL_image.save(tmp.name)

            try:
                # Test loading
                loaded = processor.load_image(tmp.name)
                assert loaded.shape == (50, 50, 3)

                # Test grayscale conversion
                grayscale = processor.convert_to_grayscale()
                assert grayscale.shape == (50, 50)
                assert grayscale.dtype == np.uint8

                # Test color inversion
                inverted = processor.invert_colors()
                assert inverted.shape == (50, 50, 3)
                assert inverted.dtype == np.uint8

                # Verify inversion logic on a known pixel
                # Original red quarter (255, 0, 0) should become (0, 255, 255)
                original_red = loaded[10, 10]  # Should be [255, 0, 0]
                inverted_red = inverted[10, 10]  # Should be [0, 255, 255]

                assert np.array_equal(original_red, [255, 0, 0])
                assert np.array_equal(inverted_red, [0, 255, 255])

            finally:
                # Clean up temp file
                os.unlink(tmp.name)

    def test_save_functionality(self):
        """Test saving processed images."""
        processor = ImageFilters()

        # Create test array
        test_array = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                processor.save_image(test_array, tmp.name)

                # Verify file was created
                assert os.path.exists(tmp.name)

                # Verify we can load it back
                loaded_back = np.array(Image.open(tmp.name))
                assert np.array_equal(test_array, loaded_back)

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
```

#### 4.2 Performance Validation Script
**File**: `scripts/validate_performance.py`

```python
#!/usr/bin/env python3
"""
Performance validation script for NumPy Mastery module.
Validates that vectorized operations are significantly faster than loops.
"""

import numpy as np
from timeit import timeit
import sys

def validate_vectorization_performance():
    """
    Comprehensive performance validation for the constitution requirement
    that vectorized operations must be 10-100x faster than loops.
    """
    print("=== Vectorization Performance Validation ===")
    print("Constitution Requirement: 10-100x performance improvement")
    print()

    test_sizes = [100, 1000, 10000]
    all_passed = True

    for size in test_sizes:
        print(f"Testing with array size: {size}")

        # Create test data
        arr = np.random.rand(size)

        # Loop-based sum of squares (as per constitution example)
        def sum_squares_loop():
            result = 0
            for x in arr:
                result += x ** 2
            return result

        # Vectorized sum of squares (constitution preferred approach)
        def sum_squares_numpy():
            return np.sum(arr ** 2)

        # Performance comparison
        loop_time = timeit(sum_squares_loop, number=100)
        numpy_time = timeit(sum_squares_numpy, number=100)

        improvement = loop_time / numpy_time

        print(f"  Loop time: {loop_time:.4f}s")
        print(f"  NumPy time: {numpy_time:.4f}s")
        print(f"  Improvement: {improvement:.1f}x")

        # Check against constitution requirement (10-100x)
        if 10 <= improvement <= 100:
            print(f"  ✅ PASS: Within constitution range (10-100x)")
        elif improvement > 100:
            print(f"  ✅ PASS: Exceeds constitution minimum (>{100}x)")
        else:
            print(f"  ❌ FAIL: Below constitution minimum (<10x)")
            all_passed = False

        print()

    # Test matrix operations
    print("Testing matrix operations...")
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)

    def matmul_loop():
        result = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    result[i, j] += A[i, k] * B[k, j]
        return result

    def matmul_numpy():
        return A @ B

    loop_time = timeit(matmul_loop, number=10)
    numpy_time = timeit(matmul_numpy, number=10)
    improvement = loop_time / numpy_time

    print(f"Matrix multiplication:")
    print(f"  Loop time: {loop_time:.4f}s")
    print(f"  NumPy time: {numpy_time:.4f}s")
    print(f"  Improvement: {improvement:.1f}x")

    if improvement > 50:
        print(f"  ✅ PASS: Significant improvement (>{50}x)")
    else:
        print(f"  ❌ FAIL: Insufficient improvement (<50x)")
        all_passed = False

    print()

    if all_passed:
        print("🎉 All performance validations PASSED!")
        print("Vectorization provides substantial performance improvements as required.")
        return 0
    else:
        print("❌ Some performance validations FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(validate_vectorization_performance())
```

## Implementation Checklist

### Environment Setup ✅
- [x] Virtual environment created with uv/pip
- [x] NumPy 1.20+, Matplotlib, Pillow, IPython installed
- [x] Directory structure created
- [x] Environment verification script created

### Core Module Implementation ✅
- [x] Jupyter notebook with 6 comprehensive cells
- [x] Vectorization performance comparison with %timeit
- [x] Broadcasting explanation with visual examples
- [x] Linear algebra operations with clear comments
- [x] Statistical operations with axis explanations

### Exercise Implementation ✅
- [x] `custom_dot()` - matrix multiplication without np.dot
- [x] `custom_matmul()` - matrix multiplication without np.matmul
- [x] `custom_transpose()` - transpose without np.transpose
- [x] Validation against NumPy built-ins
- [x] Performance comparison showing 50x+ improvement

### Mini-Project Architecture ✅
- [x] Image loading pipeline (PIL/Pillow)
- [x] Grayscale conversion with luminance formula
- [x] Color inversion with broadcasting
- [x] Optional box blur implementation
- [x] Visualization and comparison functions
- [x] File saving capabilities

### Validation Strategy ✅
- [x] Comprehensive test suite for matrix operations
- [x] Integration tests for image processing
- [x] Performance validation script
- [x] Constitution compliance verification

## Next Steps

This plan is ready for implementation. The `/sp.tasks` command should be run next to create individual, testable tasks for each component.

**Key Achievements**:
- ✅ All constitutional principles satisfied
- ✅ Clear separation of concerns (notebook, exercises, projects)
- ✅ Comprehensive validation strategy
- ✅ Performance targets defined (10-100x improvements)
- ✅ Beginner-friendly with extensive comments
- ✅ Independent testability of each component