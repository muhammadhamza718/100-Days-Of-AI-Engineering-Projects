# Feature Specification: NumPy Mastery Learning Module

**Feature Branch**: `001-numpy-mastery`
**Created**: 2026-01-15
**Status**: Draft
**Input**: User description: "Project Specifications: Day 1-2 NumPy Mastery"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Learn Core NumPy Array Operations (Priority: P1)

A data science student wants to understand NumPy fundamentals including array creation, manipulation, and performance optimization. They need clear examples that demonstrate the difference between vectorized operations and traditional loops.

**Why this priority**: This is the foundational learning objective. Without understanding core array operations, students cannot proceed to more advanced topics like linear algebra or image processing.

**Independent Test**: This can be fully tested by running the provided code examples and verifying that vectorized operations produce identical results to loop-based approaches but with significantly better performance.

**Acceptance Scenarios**:

1. **Given** a student has Python and NumPy installed, **When** they execute the array operations script, **Then** they should see examples of array creation, indexing, and basic arithmetic operations.
2. **Given** the sum-of-squares exercise, **When** they compare `sum_squares_numpy()` vs `sum_squares_loop()`, **Then** they should observe equivalent results with 10-100x performance improvement using vectorization.
3. **Given** arrays of different shapes, **When** they perform arithmetic operations, **Then** they should understand broadcasting behavior through clear examples and visualizations.

---

### User Story 2 - Implement Matrix Operations from Scratch (Priority: P2)

A student wants to understand the mathematical mechanics behind matrix operations by implementing them manually without using NumPy's built-in `np.dot` or `np.matmul` functions.

**Why this priority**: This exercise builds deep understanding of linear algebra fundamentals, which is crucial for comprehending how machine learning algorithms work under the hood.

**Independent Test**: This can be tested by verifying that custom implementations produce identical results to NumPy's built-in functions, with performance comparisons highlighting the efficiency of optimized implementations.

**Acceptance Scenarios**:

1. **Given** two 2D arrays, **When** `custom_matmul(A, B)` is called, **Then** it returns the correct matrix product matching `np.matmul(A, B)` exactly.
2. **Given** two 1D arrays, **When** `custom_dot(a, b)` is called, **Then** it returns the correct dot product matching `np.dot(a, b)` exactly.
3. **Given** a 2D array, **When** `custom_transpose(A)` is called, **Then** it returns the correctly transposed array matching `A.T` exactly.
4. **Given** all three custom functions, **When** performance is measured with `%timeit`, **Then** the educational implementations should be significantly slower than NumPy's optimized versions.

---

### User Story 3 - Build Image Processing Mini-Project (Priority: P3)

A student wants to apply NumPy array manipulation skills to process real images, understanding how images are represented as 3D arrays and how mathematical operations can create visual filters.

**Why this priority**: This project connects abstract array operations to tangible, visual results, reinforcing the concept that "everything is an array" in NumPy and providing practical experience.

**Independent Test**: This can be tested by loading an image, applying each filter, and verifying the output images match expected mathematical transformations.

**Acceptance Scenarios**:

1. **Given** a color image file (JPEG/PNG), **When** loaded using PIL/ImageIO, **Then** it should be represented as a 3D NumPy array with shape (Height × Width × 3 Channels).
2. **Given** a color image array, **When** grayscale conversion is applied using the formula Y = 0.2989R + 0.5870G + 0.1140B, **Then** the resulting image should be 2D (grayscale) and visually correct.
3. **Given** any image array, **When** color inversion is applied using 255 - pixel_value, **Then** the resulting image should show correct color inversion for all channels.
4. **Given** an image array (optional advanced), **When** box blur is applied, **Then** the image should show smooth averaging of neighboring pixels.

---

### Edge Cases

- What happens when attempting matrix multiplication on incompatible array dimensions (e.g., 2×3 matrix × 3×2 matrix is valid, but 2×3 × 2×3 should fail gracefully)?
- How does the grayscale conversion formula handle images with different channel orders (RGB vs BGR) or alpha channels (RGBA)?
- What should happen when image files are corrupted or in unsupported formats during loading?
- How should the custom matrix operations handle empty arrays or arrays with zero dimensions?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide clear, executable Python code examples demonstrating fundamental NumPy array operations (creation, indexing, slicing, shapes).
- **FR-002**: System MUST demonstrate vectorization performance benefits by showing equivalent "sum of squares" calculations using both loop-based and NumPy-optimized approaches.
- **FR-003**: System MUST provide detailed explanations of broadcasting rules with practical examples showing arithmetic between differently-shaped arrays.
- **FR-004**: System MUST implement educational versions of core linear algebra operations (`custom_matmul`, `custom_dot`, `custom_transpose`) without using `np.dot` or `np.matmul`.
- **FR-005**: System MUST include performance comparison using `%timeit` to quantify the performance difference between educational and optimized implementations.
- **FR-006**: System MUST demonstrate common statistical operations (mean, std, max, argmax) with axis-specific calculations.
- **FR-007**: System MUST provide random sampling examples using various distributions (uniform, normal, integers).
- **FR-008**: System MUST include image loading functionality using PIL or ImageIO to read common image formats (JPEG, PNG).
- **FR-009**: System MUST implement grayscale conversion using the specified luminance formula (Y = 0.2989R + 0.5870G + 0.1140B).
- **FR-010**: System MUST implement color inversion using the formula `255 - pixel_value` for all color channels.
- **FR-011**: System SHOULD optionally implement a box blur filter using pixel averaging techniques.
- **FR-012**: System MUST include visual comparison of original vs. filtered images using Matplotlib.
- **FR-013**: System MUST include comprehensive comments explaining WHY specific NumPy functions are used, particularly for axis operations.

### Key Entities

- **NumPy Array**: The fundamental data structure representing homogeneous multidimensional data, with attributes including shape, dtype, and size.
- **Image Array**: A 3D NumPy array representing color images with dimensions (Height × Width × Channels), where channels typically represent RGB values.
- **Performance Metric**: Quantitative measurement of execution time comparing vectorized vs. loop-based operations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Students can successfully execute all code examples in a Jupyter Notebook environment with Python 3.8+ and NumPy 1.20+.
- **SC-002**: Vectorized NumPy operations demonstrate 10-100x performance improvement over equivalent Python loops for sum-of-squares calculation.
- **SC-003**: All custom matrix operations (`custom_matmul`, `custom_dot`, `custom_transpose`) produce mathematically identical results to NumPy's built-in functions (verified through automated testing).
- **SC-004**: Educational implementations run at least 50x slower than NumPy's optimized versions, clearly demonstrating the value of vectorization.
- **SC-005**: Image processing mini-project successfully loads, processes, and displays at least 3 different filtered versions of an input image (grayscale, inverted, and optional blur).
- **SC-006**: All code examples include explanatory comments that help students understand not just WHAT the code does, but WHY specific NumPy functions were chosen.
- **SC-007**: Students can complete the entire learning module (exercises + mini-project) within 4-6 hours of focused study time.
- **SC-008**: Code examples are compatible with both Jupyter Notebook and VS Code Python environments without modification.