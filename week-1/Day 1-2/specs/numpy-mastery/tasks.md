# Task Breakdown: NumPy Mastery Module

**Branch**: `001-numpy-mastery` | **Date**: 2026-01-15
**Input**: Implementation Plan from `specs/numpy-mastery/plan.md`
**Constitution**: NumPy AI Engineering Learning Platform Constitution v1.0.0

**Task Status Key**:
- ⏳ **Pending**: Ready to be started
- 🔄 **In Progress**: Currently being worked on
- ✅ **Completed**: Task finished successfully
- ❌ **Blocked**: Requires external dependency

---

## Phase 1: Environment & Setup

### Task 1.1: Create Project Directory Structure
**Category**: Environment & Setup
**Priority**: P1 (Blocking)
**Estimated Time**: 5 minutes
**Dependencies**: None

**Description**: Create the complete directory structure for the NumPy Mastery project.

**Acceptance Criteria**:
- [ ] Directory `src/` exists
- [ ] Subdirectories `src/notebooks/`, `src/exercises/`, `src/projects/`, `src/utils/` exist
- [ ] Directory `tests/` exists
- [ ] Directory `data/` with subdirectories `sample_images/` and `generated/` exists
- [ ] Directory `scripts/` exists
- [ ] All `__init__.py` files created in Python packages

**Implementation Steps**:
```bash
mkdir -p src/{notebooks,exercises,projects,utils}
mkdir -p tests
mkdir -p data/{sample_images,generated}
mkdir -p scripts
touch src/__init__.py
touch src/utils/__init__.py
```

**Testing**: Verify all directories exist using `ls -la` and confirm structure matches specification.

---

### Task 1.2: Initialize Virtual Environment with uv
**Category**: Environment & Setup
**Priority**: P1 (Blocking)
**Estimated Time**: 10 minutes
**Dependencies**: Task 1.1

**Description**: Set up isolated Python environment using uv package manager.

**Acceptance Criteria**:
- [ ] uv is available on system (or fallback to pip)
- [ ] Virtual environment created with Python 3.9
- [ ] Environment activated successfully
- [ ] `python --version` shows Python 3.9+

**Implementation Steps**:
```bash
# Option A: Using uv (preferred)
uv venv --python python3.9
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Option B: Fallback to pip
python3.9 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**Testing**: Run `python --version` and `which python` to verify correct environment.

---

### Task 1.3: Install Core Dependencies
**Category**: Environment & Setup
**Priority**: P1 (Blocking)
**Estimated Time**: 5 minutes
**Dependencies**: Task 1.2

**Description**: Install all required Python packages with specific versions.

**Acceptance Criteria**:
- [ ] NumPy 1.24.0 installed
- [ ] Matplotlib 3.7.0 installed
- [ ] Pillow 9.4.0 installed
- [ ] IPython and Jupyter installed
- [ ] All packages import without errors

**Implementation Steps**:
```bash
# Using uv (preferred)
uv pip install numpy==1.24.0 matplotlib==3.7.0 pillow==9.4.0 ipython jupyter

# Fallback to pip
pip install numpy==1.24.0 matplotlib==3.7.0 pillow==9.4.0 ipython jupyter
```

**Testing**:
```python
# verification script
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IPython
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Pillow: {Image.__version__}")
```

---

### Task 1.4: Create Environment Verification Script
**Category**: Environment & Setup
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 1.3

**Description**: Create automated script to verify environment setup.

**Acceptance Criteria**:
- [ ] Script `src/utils/verify_environment.py` created
- [ ] Script runs without errors
- [ ] Script outputs all package versions
- [ ] Script validates NumPy version >= 1.20.0

**Implementation**:
```python
# src/utils/verify_environment.py
import numpy as np
import matplotlib
from PIL import Image
import sys

def verify_environment():
    """Verify all required packages are installed with correct versions."""
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Pillow: {Image.__version__}")

    # Verify NumPy version requirement
    assert np.__version__ >= "1.20.0", f"NumPy 1.20+ required, got {np.__version__}"
    print("✅ Environment verified successfully!")

if __name__ == "__main__":
    verify_environment()
```

**Testing**: Run `python src/utils/verify_environment.py` and confirm all versions are displayed.

---

## Phase 2: Core Concepts Notebook

### Task 2.1: Create Main Jupyter Notebook Structure
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 5 minutes
**Dependencies**: Task 1.3

**Description**: Create the main educational notebook with basic structure and imports.

**Acceptance Criteria**:
- [ ] File `src/notebooks/numpy_fundamentals.ipynb` created
- [ ] First cell contains all required imports
- [ ] Notebook runs without import errors
- [ ] NumPy version is displayed

**Implementation**:
```python
# Cell 1: Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import sys

print("NumPy Mastery - Day 1-2 Learning Module")
print(f"NumPy version: {np.__version__}")
```

**Testing**: Open notebook in Jupyter and run first cell. Verify no import errors.

---

### Task 2.2: Add Array Creation and Operations Examples
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 2.1

**Description**: Add comprehensive examples of array creation and basic operations.

**Acceptance Criteria**:
- [ ] Cell 2 created with array creation examples
- [ ] Demonstrates 1D, 2D, and 3D array creation
- [ ] Shows shape attributes for each array
- [ ] Includes element-wise arithmetic operations
- [ ] All code runs without errors

**Implementation**:
```python
# Cell 2: Array Creation and Basic Operations
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

**Testing**: Run cell and verify all shapes are correct and arithmetic works as expected.

---

### Task 2.3: Implement Vectorization Performance Comparison
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 15 minutes
**Dependencies**: Task 2.2

**Description**: Create performance comparison between loop-based and vectorized sum of squares.

**Acceptance Criteria**:
- [ ] Cell 3 created with both loop and NumPy implementations
- [ ] Function `sum_squares_loop()` implemented correctly
- [ ] Function `sum_squares_numpy()` implemented correctly
- [ ] Performance comparison uses `%timeit` or `timeit`
- [ ] Output shows 10-100x performance improvement
- [ ] Comments explain WHY vectorization is faster

**Implementation**:
```python
# Cell 3: Vectorization Performance Comparison
def sum_squares_loop(arr):
    """Non-vectorized implementation (slow) - for educational comparison only."""
    result = 0
    for x in arr:
        result += x ** 2
    return result

def sum_squares_numpy(arr):
    """Vectorized implementation (fast) - primary approach."""
    return np.sum(arr ** 2)  # WHY: np.sum operates at C speed, optimized for arrays

# Performance comparison
test_array = np.random.rand(10000)
loop_time = timeit(lambda: sum_squares_loop(test_array), number=100)
numpy_time = timeit(lambda: sum_squares_numpy(test_array), number=100)

print(f"Loop implementation: {loop_time:.4f}s")
print(f"NumPy implementation: {numpy_time:.4f}s")
print(f"Performance improvement: {loop_time/numpy_time:.1f}x faster")
```

**Testing**: Run cell and verify NumPy implementation is 10-100x faster than loop version.

---

### Task 2.4: Add Broadcasting Examples with Explanation
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 2.3

**Description**: Demonstrate broadcasting concept with clear explanation and examples.

**Acceptance Criteria**:
- [ ] Cell 4 created with broadcasting examples
- [ ] Shows arrays with different shapes
- [ ] Demonstrates broadcasting from (3,) to (2, 3)
- [ ] Includes explanation using "stretching" analogy
- [ ] Visual representation of broadcasted array

**Implementation**:
```python
# Cell 4: Broadcasting Explanation
# Broadcasting example - "stretching" smaller array to match larger
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
b = np.array([10, 20, 30])  # Shape: (3,)

print("Array A shape:", A.shape)
print("Array b shape:", b.shape)

# b is "stretched" to shape (1, 3) then (2, 3) for element-wise addition
result = A + b
print("Broadcasting result:")
print(result)
print("Result shape:", result.shape)

# Explanation: b is conceptually repeated along axis 0:
# [[10, 20, 30],  <- original b
#  [10, 20, 30]]  <- broadcasted b
```

**Testing**: Run cell and verify the broadcasting produces expected result and explanation is clear.

---

### Task 2.5: Add Linear Algebra Operations Examples
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 2.4

**Description**: Demonstrate basic linear algebra operations including dot products and matrix multiplication.

**Acceptance Criteria**:
- [ ] Cell 5 created with linear algebra examples
- [ ] Dot product example with clear calculation
- [ ] Matrix multiplication using `@` operator
- [ ] Additional operations (inverse, eigenvalues, transpose)
- [ ] All examples run without errors

**Implementation**:
```python
# Cell 5: Linear Algebra Operations
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

**Testing**: Run cell and verify all linear algebra operations produce correct results.

---

### Task 2.6: Add Random Sampling and Statistics Examples
**Category**: Core Concepts
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 2.5

**Description**: Demonstrate random sampling and statistical operations with axis-specific calculations.

**Acceptance Criteria**:
- [ ] Cell 6 created with random sampling examples
- [ ] Shows uniform, normal, and integer random generation
- [ ] Demonstrates statistical functions (mean, std, max, argmax)
- [ ] Includes axis-specific operations with clear explanations
- [ ] Comments explain WHY axis=0 vs axis=1

**Implementation**:
```python
# Cell 6: Random Sampling and Statistics
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

**Testing**: Run cell and verify statistical calculations and axis-specific operations are correct.

---

## Phase 3: From-Scratch Matrix Operations

### Task 3.1: Create exercises.py File Structure
**Category**: From-Scratch Exercises
**Priority**: P1
**Estimated Time**: 5 minutes
**Dependencies**: Task 1.1

**Description**: Create the main exercises file with imports and function stubs.

**Acceptance Criteria**:
- [ ] File `src/exercises/matrix_operations.py` created
- [ ] All required imports included
- [ ] Function stubs created for all three operations
- [ ] File runs without syntax errors

**Implementation**:
```python
# src/exercises/matrix_operations.py
import numpy as np
from timeit import timeit

def custom_dot(a, b):
    """Custom dot product implementation using loops only."""
    pass

def custom_matmul(A, B):
    """Custom matrix multiplication using loops only."""
    pass

def custom_transpose(A):
    """Custom matrix transpose using loops only."""
    pass

if __name__ == "__main__":
    print("Matrix Operations Exercises - Ready for implementation")
```

**Testing**: Run `python src/exercises/matrix_operations.py` and verify no errors.

---

### Task 3.2: Implement custom_dot() Function
**Category**: From-Scratch Exercises
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 3.1

**Description**: Implement dot product using only loops and array indexing.

**Acceptance Criteria**:
- [ ] Function accepts two 1D NumPy arrays
- [ ] Validates array lengths match
- [ ] Uses only Python loops and indexing (no np.dot)
- [ ] Returns correct dot product value
- [ ] Includes comprehensive docstring

**Implementation**:
```python
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
```

**Testing**: Verify with `custom_dot(np.array([1,2,3]), np.array([4,5,6])) == 32`

---

### Task 3.3: Implement custom_matmul() Function
**Category**: From-Scratch Exercises
**Priority**: P1
**Estimated Time**: 15 minutes
**Dependencies**: Task 3.2

**Description**: Implement matrix multiplication using only nested loops and array indexing.

**Acceptance Criteria**:
- [ ] Function accepts two 2D NumPy arrays
- [ ] Validates inner dimensions match
- [ ] Uses only Python loops and indexing (no np.matmul/@)
- [ ] Returns correct matrix product
- [ ] Includes comprehensive docstring

**Implementation**:
```python
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
```

**Testing**: Verify with `custom_matmul(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))` matches expected result.

---

### Task 3.4: Implement custom_transpose() Function
**Category**: From-Scratch Exercises
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 3.3

**Description**: Implement matrix transpose using only loops and array indexing.

**Acceptance Criteria**:
- [ ] Function accepts 2D NumPy array
- [ ] Validates input is 2D (raises error otherwise)
- [ ] Uses only Python loops and indexing (no np.transpose/.T)
- [ ] Returns correctly transposed array
- [ ] Includes comprehensive docstring

**Implementation**:
```python
def custom_transpose(A):
    """
    Custom matrix transpose using loops only.

    Args:
        A (np.ndarray): 2D array

    Returns:
        np.ndarray: Transposed array

    Raises:
        ValueError: If array is not 2D
    """
    if len(A.shape) != 2:
        raise ValueError("Transpose only supported for 2D arrays")

    rows, cols = A.shape
    result = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j, i] = A[i, j]

    return result
```

**Testing**: Verify with `custom_transpose(np.array([[1,2,3],[4,5,6]]))` matches `np.array([[1,4],[2,5],[3,6]])`

---

### Task 3.5: Create Validation and Comparison Functions
**Category**: From-Scratch Exercises
**Priority**: P1
**Estimated Time**: 15 minutes
**Dependencies**: Task 3.4

**Description**: Create functions to validate custom implementations against NumPy built-ins.

**Acceptance Criteria**:
- [ ] `validate_custom_implementations()` function created
- [ ] Tests dot product with known values
- [ ] Tests matrix multiplication with 2D arrays
- [ ] Tests transpose operation
- [ ] Uses `np.isclose()` and `np.allclose()` for floating point comparison
- [ ] Function runs and produces clear output

**Implementation**:
```python
def validate_custom_implementations():
    """Validate custom implementations against NumPy built-ins."""
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
```

**Testing**: Run `validate_custom_implementations()` and verify all tests pass.

---

### Task 3.6: Create Performance Comparison Function
**Category**: From-Scratch Exercises
**Priority**: P2
**Estimated Time**: 10 minutes
**Dependencies**: Task 3.5

**Description**: Create function to compare performance between custom and NumPy implementations.

**Acceptance Criteria**:
- [ ] `performance_comparison()` function created
- [ ] Uses `timeit` for accurate timing
- [ ] Tests with large arrays for meaningful comparison
- [ ] Shows quantitative performance ratios
- [ ] Function runs and produces clear output

**Implementation**:
```python
def performance_comparison():
    """Compare performance between custom and NumPy implementations."""
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
```

**Testing**: Run `performance_comparison()` and verify NumPy is at least 50x faster.

---

### Task 3.7: Add Main Execution Block
**Category**: From-Scratch Exercises
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 3.6

**Description**: Add `if __name__ == "__main__"` block to run validation and performance tests.

**Acceptance Criteria**:
- [ ] Main block calls validation function
- [ ] Main block calls performance comparison function
- [ ] Script runs end-to-end without errors
- [ ] Output is clear and informative

**Implementation**:
```python
if __name__ == "__main__":
    validate_custom_implementations()
    performance_comparison()
```

**Testing**: Run `python src/exercises/matrix_operations.py` and verify complete execution.

---

## Phase 4: Image Processing Mini-Project

### Task 4.1: Create image_filters.py File Structure
**Category**: Mini-Project
**Priority**: P1
**Estimated Time**: 5 minutes
**Dependencies**: Task 1.1

**Description**: Create main image processing file with class structure and basic imports.

**Acceptance Criteria**:
- [ ] File `src/projects/image_filters.py` created
- [ ] `ImageFilters` class defined
- [ ] All required imports included
- [ ] Basic constructor implemented
- [ ] File runs without syntax errors

**Implementation**:
```python
# src/projects/image_filters.py
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

if __name__ == "__main__":
    print("Image Filters Mini-Project - Ready for implementation")
```

**Testing**: Run `python src/projects/image_filters.py` and verify no errors.

---

### Task 4.2: Implement Image Loading Method
**Category**: Mini-Project
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 4.1

**Description**: Implement method to load images from file and convert to NumPy arrays.

**Acceptance Criteria**:
- [ ] `load_image()` method implemented
- [ ] Handles file existence check
- [ ] Converts images to RGB format (handles RGBA/grayscale)
- [ ] Stores image as NumPy array
- [ ] Prints image information (shape, dtype, value range)
- [ ] Returns the image array

**Implementation**:
```python
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
```

**Testing**: Use a test image to verify loading works correctly.

---

### Task 4.3: Implement Grayscale Conversion Method
**Category**: Mini-Project
**Priority**: P1
**Estimated Time**: 15 minutes
**Dependencies**: Task 4.2

**Description**: Implement vectorized grayscale conversion using luminance formula.

**Acceptance Criteria**:
- [ ] `convert_to_grayscale()` method implemented
- [ ] Uses luminance formula: Y = 0.2989R + 0.5870G + 0.1140B
- [ ] Uses array slicing and broadcasting (no loops)
- [ ] Returns 2D array (grayscale)
- [ ] Includes explanation of WHY the formula is used
- [ ] Handles input validation

**Implementation**:
```python
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
```

**Testing**: Test with a known RGB image and verify grayscale conversion produces expected results.

---

### Task 4.4: Implement Color Inversion Method
**Category**: Mini-Project
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 4.3

**Description**: Implement vectorized color inversion using broadcasting.

**Acceptance Criteria**:
- [ ] `invert_colors()` method implemented
- [ ] Uses formula: New_Pixel = 255 - Old_Pixel
- [ ] Uses broadcasting (no loops)
- [ ] Returns array of same shape as input
- [ ] Includes explanation of the operation

**Implementation**:
```python
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
```

**Testing**: Test with a known image and verify each RGB channel is correctly inverted.

---

### Task 4.5: Create Test Image Generation Function
**Category**: Mini-Project
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 4.4

**Description**: Create helper function to generate test images for demonstration.

**Acceptance Criteria**:
- [ ] `create_test_image()` function created
- [ ] Generates colorful test pattern
- [ ] Creates distinct color regions (red, green, blue, gray)
- [ ] Returns valid NumPy array image

**Implementation**:
```python
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
```

**Testing**: Run function and verify it creates a valid colorful image array.

---

### Task 4.6: Create Display Comparison Method
**Category**: Mini-Project
**Priority**: P2
**Estimated Time**: 10 minutes
**Dependencies**: Task 4.5

**Description**: Implement method to display original and filtered images side by side.

**Acceptance Criteria**:
- [ ] `display_comparison()` method implemented
- [ ] Creates side-by-side subplot
- [ ] Handles both color and grayscale images
- [ ] Includes proper titles and removes axes
- [ ] Uses appropriate colormap for grayscale

**Implementation**:
```python
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
```

**Testing**: Test with known images to verify proper display and handling of both color/grayscale.

---

### Task 4.7: Create Save Image Method
**Category**: Mini-Project
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 4.6

**Description**: Implement method to save NumPy array as image file.

**Acceptance Criteria**:
- [ ] `save_image()` method implemented
- [ ] Handles data type conversion to uint8
- [ ] Handles both color and grayscale images
- [ ] Saves file successfully
- [ ] Prints confirmation message

**Implementation**:
```python
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
```

**Testing**: Save a test image and verify it can be loaded back correctly.

---

### Task 4.8: Create Demonstration Function
**Category**: Mini-Project
**Priority**: P1
**Estimated Time**: 10 minutes
**Dependencies**: Task 4.7

**Description**: Create main demonstration function that showcases the complete pipeline.

**Acceptance Criteria**:
- [ ] `demonstrate_mini_project()` function created
- [ ] Uses test image generation
- [ ] Demonstrates grayscale conversion
- [ ] Demonstrates color inversion
- [ ] Shows side-by-side comparisons
- [ ] Includes clear console output
- [ ] Runs without errors

**Implementation**:
```python
def demonstrate_mini_project():
    """
    Main demonstration function for the image processing mini-project.
    """
    print("=== Image Processing Mini-Project ===")
    print("This demonstrates that images are 3D NumPy arrays (H × W × C)")
    print("and filters are mathematical operations on these arrays.\n")

    # Initialize processor
    processor = ImageFilters()

    # Use test image (in practice, use: processor.load_image("path/to/image.jpg"))
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

    print("\n=== Mini-Project Complete ===")
    print("Key takeaway: All operations were mathematical transformations")
    print("on the NumPy array representation of the image!")
```

**Testing**: Run `demonstrate_mini_project()` and verify complete pipeline works.

---

### Task 4.9: Add Main Execution Block
**Category**: Mini-Project
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 4.8

**Description**: Add main execution block to run demonstration when script is executed directly.

**Acceptance Criteria**:
- [ ] `if __name__ == "__main__":` block added
- [ ] Calls demonstration function
- [ ] Script runs end-to-end without errors
- [ ] Displays visualizations (may require GUI backend)

**Implementation**:
```python
if __name__ == "__main__":
    demonstrate_mini_project()
```

**Testing**: Run `python src/projects/image_filters.py` and verify visualization appears.

---

## Phase 5: Validation & Testing

### Task 5.1: Create Unit Tests for Matrix Operations
**Category**: Validation
**Priority**: P1
**Estimated Time**: 20 minutes
**Dependencies**: Task 3.7

**Description**: Create comprehensive test suite for custom matrix operations.

**Acceptance Criteria**:
- [ ] File `tests/test_matrix_operations.py` created
- [ ] Tests for `custom_dot()` with various inputs
- [ ] Tests for `custom_matmul()` with different shapes
- [ ] Tests for `custom_transpose()` with edge cases
- [ ] Tests error handling for invalid inputs
- [ ] Tests performance validation
- [ ] All tests pass when run

**Implementation**:
```python
# tests/test_matrix_operations.py
import numpy as np
import pytest
from src.exercises.matrix_operations import custom_dot, custom_matmul, custom_transpose

class TestMatrixOperations:
    def test_custom_dot_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        custom_result = custom_dot(a, b)
        expected = np.dot(a, b)
        assert np.isclose(custom_result, expected)

    def test_custom_matmul_basic(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        custom_result = custom_matmul(A, B)
        expected = A @ B
        assert np.allclose(custom_result, expected)

    def test_custom_transpose_2d(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        custom_result = custom_transpose(A)
        expected = A.T
        assert np.allclose(custom_result, expected)

    def test_custom_dot_error(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2])
        with pytest.raises(ValueError):
            custom_dot(a, b)

    def test_custom_matmul_error(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            custom_matmul(A, B)
```

**Testing**: Run `pytest tests/test_matrix_operations.py` and verify all tests pass.

---

### Task 5.2: Create Unit Tests for Image Filters
**Category**: Validation
**Priority**: P1
**Estimated Time**: 15 minutes
**Dependencies**: Task 4.9

**Description**: Create test suite for image processing functionality.

**Acceptance Criteria**:
- [ ] File `tests/test_image_filters.py` created
- [ ] Tests grayscale conversion with known values
- [ ] Tests color inversion with predictable results
- [ ] Tests edge cases and error handling
- [ ] All tests pass when run

**Implementation**:
```python
# tests/test_image_filters.py
import numpy as np
import pytest
import tempfile
import os
from PIL import Image
from src.projects.image_filters import ImageFilters, create_test_image

class TestImageFilters:
    def test_grayscale_conversion(self):
        processor = ImageFilters()
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[:, :, 0] = 100  # Red
        test_image[:, :, 1] = 150  # Green
        test_image[:, :, 2] = 200  # Blue

        processor.image_array = test_image
        grayscale = processor.convert_to_grayscale()

        expected_value = 0.2989 * 100 + 0.5870 * 150 + 0.1140 * 200
        assert grayscale.shape == (10, 10)
        assert np.allclose(grayscale, expected_value, atol=1)

    def test_color_inversion(self):
        processor = ImageFilters()
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        processor.image_array = test_image

        inverted = processor.invert_colors()
        expected = np.array([[[155, 105, 55]]], dtype=np.uint8)
        assert np.array_equal(inverted, expected)

    def test_create_test_image(self):
        image = create_test_image()
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8
```

**Testing**: Run `pytest tests/test_image_filters.py` and verify all tests pass.

---

### Task 5.3: Create Performance Validation Script
**Category**: Validation
**Priority**: P2
**Estimated Time**: 10 minutes
**Dependencies**: Task 3.7

**Description**: Create standalone script to validate performance improvements meet constitution requirements.

**Acceptance Criteria**:
- [ ] File `scripts/validate_performance.py` created
- [ ] Tests vectorization vs loops for sum of squares
- [ ] Tests matrix multiplication performance
- [ ] Validates 10-100x improvement requirement
- [ ] Provides clear pass/fail output

**Implementation**:
```python
# scripts/validate_performance.py
import numpy as np
from timeit import timeit
import sys

def validate_vectorization_performance():
    """Validate vectorization provides 10-100x improvement."""
    print("=== Vectorization Performance Validation ===")

    test_sizes = [100, 1000, 10000]
    all_passed = True

    for size in test_sizes:
        print(f"Testing with array size: {size}")
        arr = np.random.rand(size)

        def sum_squares_loop():
            result = 0
            for x in arr:
                result += x ** 2
            return result

        def sum_squares_numpy():
            return np.sum(arr ** 2)

        loop_time = timeit(sum_squares_loop, number=100)
        numpy_time = timeit(sum_squares_numpy, number=100)
        improvement = loop_time / numpy_time

        print(f"  Improvement: {improvement:.1f}x")

        if improvement >= 10:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
            all_passed = False

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(validate_vectorization_performance())
```

**Testing**: Run `python scripts/validate_performance.py` and verify it passes.

---

### Task 5.4: Create Complete Test Runner
**Category**: Validation
**Priority**: P2
**Estimated Time**: 5 minutes
**Dependencies**: Task 5.1, 5.2, 5.3

**Description**: Create script to run all tests and validations in sequence.

**Acceptance Criteria**:
- [ ] Script `scripts/run_all_tests.py` created
- [ ] Runs unit tests for matrix operations
- [ ] Runs unit tests for image filters
- [ ] Runs performance validation
- [ ] Provides summary of results
- [ ] Returns appropriate exit code

**Implementation**:
```python
# scripts/run_all_tests.py
import subprocess
import sys

def run_all_tests():
    """Run all tests and validations."""
    print("🚀 Running Complete Test Suite for NumPy Mastery Module")
    print("=" * 60)

    tests = [
        ("Unit Tests: Matrix Operations", ["pytest", "tests/test_matrix_operations.py", "-v"]),
        ("Unit Tests: Image Filters", ["pytest", "tests/test_image_filters.py", "-v"]),
        ("Performance Validation", ["python", "scripts/validate_performance.py"]),
    ]

    all_passed = True

    for test_name, command in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode != 0:
                print(f"❌ {test_name} FAILED")
                all_passed = False
            else:
                print(f"✅ {test_name} PASSED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Module is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please review.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
```

**Testing**: Run `python scripts/run_all_tests.py` and verify complete test suite passes.

---

## Task Completion Summary

### Total Tasks by Category:
- **Environment & Setup**: 4 tasks
- **Core Concepts**: 6 tasks
- **From-Scratch Exercises**: 7 tasks
- **Mini-Project**: 9 tasks
- **Validation & Testing**: 4 tasks

**Total**: 30 granular, actionable tasks

### Task Dependencies:
- **Phase 1** (Environment) must complete before **Phase 2-4**
- **Phase 2-4** can proceed in parallel once environment is ready
- **Phase 5** (Testing) depends on completion of Phases 2-4

### Estimated Total Time:
- **Core Implementation**: ~2-3 hours
- **Testing & Validation**: ~1 hour
- **Total**: ~3-4 hours (fits within 4-6 hour target from specification)

### Constitutional Compliance Check:
✅ All tasks designed to:
- Prioritize vectorized solutions (where applicable)
- Prohibit `np.dot`/`np.matmul` in educational exercises
- Explain broadcasting concepts
- Use descriptive variable names
- Emphasize images as 3D arrays
- Include performance validation

**Ready for implementation!** 🚀