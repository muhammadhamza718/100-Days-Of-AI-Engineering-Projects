# Day 1-2: NumPy Mastery

## Overview

Welcome to Days 1-2 of your AI Engineering journey! In this module, we will dive deep into **NumPy**, the fundamental package for scientific computing in Python. You will learn about efficient array manipulations, linear algebra, and statistical operations which are the building blocks of almost all machine learning algorithms.

## Tech Stack

- **Language:** Python 3.x
- **Core Library:** NumPy
- **Visualization:** Matplotlib
- **Utilities:** PIL (Python Imaging Library) or ImageIO (for image loading)
- **Package Manager:** uv
- **Environment:** Jupyter Notebook or VS Code

---

## 1. Array Operations, Broadcasting, and Vectorization

### 1.1 Array Creation and Basic Operations

NumPy's main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers.

```python
import numpy as np

# Creating arrays
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])

print(f"Shape of a: {a.shape}")
print(f"Shape of b: {b.shape}")

# Basic Arithmetic
# Operations are element-wise
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])

print(f"Add: {x + y}")        # [11 22 33]
print(f"Subtract: {x - y}")   # [ 9 18 27]
print(f"Multiply: {x * y}")   # [10 40 90] (Element-wise multiplication)
print(f"Divide: {x / y}")     # [10. 10. 10.]
```

### 1.2 Vectorization

Vectorization enables you to express batch operations on data without writing any `for` loops. This is not only more concise but also significantly faster because it leverages low-level C optimizations.

**Example: Computing the sum of squares**

_Non-Vectorized (Slow):_

```python
def sum_squares_loop(arr):
    result = 0
    for x in arr:
        result += x ** 2
    return result
```

_Vectorized (Fast):_

```python
def sum_squares_numpy(arr):
    return np.sum(arr ** 2)
```

### 1.3 Broadcasting

Broadcasting allows NumPy to work with arrays of different shapes during arithmetic operations. The smaller array is "broadcast" across the larger array so that they have compatible shapes.

**Rules of Broadcasting:**

1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
2. The two arrays are strictly compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]]) # Shape (2, 3)
b = np.array([10, 20, 30]) # Shape (3,)

# b is treated as shape (1, 3) and then broadcast to (2, 3)
result = A + b
print(result)
# Output:
# [[11 22 33]
#  [14 25 36]]
```

---

## 2. Linear Algebra Operations

Linear algebra is at the heart of Deep Learning.

### 2.1 Dot Product

The dot product is the sum of the products of corresponding entries of two sequences of numbers.

```python
a = np.array([1, 2])
b = np.array([3, 4])

dot_product = np.dot(a, b) # 1*3 + 2*4 = 11
# Or using the @ operator (Python 3.5+)
dot_product_v2 = a @ b
```

### 2.2 Matrix Multiplication

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.matmul(A, B)
# OR
D = A @ B

print(C)
# [[19 22]
#  [43 50]]
```

### 2.3 Other Common Operations

- `np.linalg.inv()`: Inverse of a matrix
- `np.linalg.eig()`: Eigenvalues and eigenvectors
- `np.linalg.norm()`: Matrix or vector norm (magnitude)
- `np.transpose()` or `arr.T`: Transpose of a matrix

---

## 3. Random Sampling and Statistical Operations

Simulating data and calculating statistics is essential for data analysis.

### 3.1 Random Sampling

```python
# Random numbers between 0 and 1
rand_uniform = np.random.rand(3, 2)

# Random integers
rand_int = np.random.randint(0, 10, size=(3, 3))

# Standard normal distribution (mean=0, std=1)
rand_normal = np.random.randn(1000)

# Random choice from an array
choices = np.random.choice([10, 20, 30, 40], size=5)
```

### 3.2 Statistical Functions

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(f"Mean: {np.mean(data)}")
print(f"Standard Deviation: {np.std(data)}")
print(f"Max: {np.max(data)}")
print(f"Argmax (index of max): {np.argmax(data)}")

# Axis-specific statistics
print(f"Sum across columns (axis=0): {np.sum(data, axis=0)}") # [5 7 9]
print(f"Sum across rows (axis=1): {np.sum(data, axis=1)}")    # [6 15]
```

---

## 4. Exercise: Implement Matrix Operations from Scratch

**Goal:** Understand the mechanics behind matrix operations by implementing them without using `np.dot` or `np.matmul` (you can use NumPy arrays for storage and indexing).

**Tasks:**

1.  **Matrix Multiplication:** Write a function `custom_matmul(A, B)` that takes two 2D arrays and returns their matrix product using loops. Compare its performance (`timeit`) with `np.matmul`.
2.  **Dot Product:** Write a function `custom_dot(a, b)` for two 1D arrays.
3.  **Transpose:** Write a function `custom_transpose(A)` that returns the transpose of a 2D array.

**Self-Check:** verify your results against NumPy's built-in functions.

---

## 5. Mini Project: Build a Simple Image Filter

**Goal:** Apply vectorization and array manipulation to process images. Images are just 3D arrays of numbers (Height, Width, Color Channels)!

**Prerequisites:**

- `numpy`
- `matplotlib.pyplot` (to view images)
- `PIL` or `imageio` (to load images)

**Tasks:**

1.  **Load an image:** Load any JPEG/PNG image into a NumPy array.
2.  **Grayscale Conversion:** Convert the color image to grayscale. A simple formula is:
    $Y = 0.2989 \times R + 0.5870 \times G + 0.1140 \times B$
    _Hint: Use broadcasting and matrix multiplication._
3.  **Inversion Filter:** Create a filter that inverts the colors (negative image).
    _Hint: For 8-bit images, `New_Pixel = 255 - Old_Pixel`._
4.  **Blur Filter (Optional/Advanced):** Implement a simple box blur by averaging the pixel values of neighbors. (You might need to learn about "convolutions" or simple slicing techniques).

**Deliverables:**

- A Python script or Jupyter Notebook showcasing original vs. filtered images.
