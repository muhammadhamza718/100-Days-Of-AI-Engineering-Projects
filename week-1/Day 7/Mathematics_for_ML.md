# Day 7: Mathematics for Machine Learning

## Overview

Welcome to Day 7 of your AI Engineering journey! Today, we build the **mathematical foundation** essential for understanding and implementing machine learning algorithms. You'll master the core concepts from **Linear Algebra**, **Calculus**, and **Probability & Statistics** that power modern ML. By the end, you'll understand gradient descent, the optimization algorithm at the heart of neural networks.

## Tech Stack

- **Language:** Python 3.x
- **Core Libraries:** NumPy
- **Visualization:** Matplotlib
- **Package Manager:** uv

---

## 1. Linear Algebra Review

Linear algebra is the language of machine learning. Vectors represent data points, matrices represent datasets, and transformations represent models.

### 1.1 Vectors

**Definition:** A vector is an ordered array of numbers representing magnitude and direction.

**Vector Operations:**

- **Addition:** Element-wise addition of corresponding components
  - v₁ + v₂ = [v₁₁ + v₂₁, v₁₂ + v₂₂, ..., v₁ₙ + v₂ₙ]
- **Scalar Multiplication:** Multiply each component by a scalar
  - c · v = [c·v₁, c·v₂, ..., c·vₙ]
- **Dot Product:** Sum of element-wise products
  - v₁ · v₂ = v₁₁·v₂₁ + v₁₂·v₂₂ + ... + v₁ₙ·v₂ₙ
  - Geometric interpretation: v₁ · v₂ = |v₁| |v₂| cos(θ)
- **Magnitude (L2 Norm):** Length of a vector
  - |v| = √(v₁² + v₂² + ... + vₙ²)
- **Unit Vector:** Vector with magnitude 1
  - v̂ = v / |v|

**Key Concepts:**

- **Dot Product:** Measures similarity between vectors (high = similar direction, zero = orthogonal)
- **Norm:** Represents the length/magnitude of a vector
- **Orthogonal Vectors:** Two vectors are orthogonal if their dot product equals zero (they are perpendicular)
- **Vector Space:** A collection of vectors that can be added together and multiplied by scalars

**Applications in ML:**

- Feature vectors represent data samples
- Similarity measures (cosine similarity uses dot product)
- Distance metrics (Euclidean distance uses L2 norm)

### 1.2 Matrices

**Definition:** A matrix is a 2D rectangular array of numbers arranged in rows and columns. In ML, each row often represents a data sample, and each column represents a feature.

**Matrix Notation:**

- A matrix A of size m×n has m rows and n columns
- Element at row i, column j is denoted as Aᵢⱼ

**Matrix Operations:**

1. **Transpose (Aᵀ):** Flip rows and columns

   - If A is m×n, then Aᵀ is n×m
   - (Aᵀ)ᵢⱼ = Aⱼᵢ

2. **Matrix Addition:** Add corresponding elements (requires same dimensions)

   - C = A + B where Cᵢⱼ = Aᵢⱼ + Bᵢⱼ

3. **Scalar Multiplication:** Multiply every element by a scalar

   - C = c·A where Cᵢⱼ = c·Aᵢⱼ

4. **Matrix Multiplication:** Combine two matrices
   - For A (m×n) and B (n×p), result C is (m×p)
   - Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ (dot product of row i of A and column j of B)
   - **NOT commutative:** A·B ≠ B·A (in general)
   - **Associative:** (A·B)·C = A·(B·C)

**Matrix Multiplication Rules:**

- Inner dimensions must match: (m×n)·(n×p) = (m×p)
- Element (i,j) = dot product of A's row i with B's column j
- Represents composition of linear transformations

### 1.3 Special Matrices

**Identity Matrix (I):**

- Square matrix with 1s on diagonal, 0s elsewhere
- Property: A·I = I·A = A
- Acts as the "multiplication identity" for matrices

**Inverse Matrix (A⁻¹):**

- Matrix that "undoes" A: A·A⁻¹ = A⁻¹·A = I
- Only exists for square, non-singular matrices
- If det(A) ≠ 0, then A is invertible

**Determinant (det(A)):**

- Scalar value that encodes certain properties of the matrix
- If det(A) = 0, matrix is singular (not invertible)
- Represents the scaling factor of the transformation
- For 2×2 matrix: det([[a,b],[c,d]]) = ad - bc

**Symmetric Matrix:**

- A = Aᵀ (equals its own transpose)
- Important in optimization (Hessian matrices)

**Diagonal Matrix:**

- All non-diagonal elements are zero
- Efficient for computation

### 1.4 Eigenvalues and Eigenvectors

**Definition:** For a square matrix A, eigenvector **v** and eigenvalue **λ** satisfy:

**A·v = λ·v**

This means applying transformation A to v only scales it by λ, without changing its direction.

**Properties:**

- A has n eigenvalues (counting multiplicities) if it's n×n
- Eigenvectors corresponding to different eigenvalues are linearly independent
- For symmetric matrices, eigenvectors are orthogonal

**Eigenvalue Decomposition:**

- A = Q·Λ·Q⁻¹
- Where Q contains eigenvectors as columns
- Λ is a diagonal matrix of eigenvalues

**Applications in ML:**

1. **Principal Component Analysis (PCA):**

   - Finds directions (eigenvectors) of maximum variance in data
   - Eigenvalues indicate the importance of each direction
   - Used for dimensionality reduction

2. **Spectral Clustering:**

   - Uses eigenvectors of similarity matrices
   - Reveals underlying cluster structure

3. **Stability Analysis:**

   - Eigenvalues determine if a system is stable
   - Used in analyzing neural network dynamics

4. **Matrix Powers:**
   - Computing Aⁿ efficiently using eigendecomposition

---

## 2. Calculus Basics

Calculus enables us to optimize functions, which is the core of training machine learning models.

### 2.1 Derivatives

**Definition:** The derivative f'(x) measures the instantaneous rate of change of function f at point x.

**Geometric Interpretation:**

- Derivative = slope of the tangent line at a point
- Positive derivative → function is increasing
- Negative derivative → function is decreasing
- Zero derivative → potential local minimum/maximum

**Fundamental Derivative Rules:**

1. **Power Rule:**

   - d/dx(xⁿ) = n·xⁿ⁻¹

2. **Constant Rule:**

   - d/dx(c) = 0

3. **Sum Rule:**

   - d/dx(f + g) = f' + g'

4. **Product Rule:**

   - d/dx(f·g) = f'·g + f·g'

5. **Quotient Rule:**

   - d/dx(f/g) = (f'·g - f·g') / g²

6. **Chain Rule:**
   - d/dx(f(g(x))) = f'(g(x))·g'(x)

**Common Derivatives:**

- d/dx(eˣ) = eˣ
- d/dx(ln(x)) = 1/x
- d/dx(sin(x)) = cos(x)
- d/dx(cos(x)) = -sin(x)

**Applications in ML:**

- Computing gradients for optimization
- Backpropagation in neural networks
- Finding optimal parameters

### 2.2 Partial Derivatives

**Definition:** For functions with multiple variables f(x₁, x₂, ..., xₙ), a partial derivative measures the rate of change with respect to one variable while holding all others constant.

**Notation:**

- ∂f/∂x₁ = partial derivative with respect to x₁
- ∂f/∂x₂ = partial derivative with respect to x₂

**Example:**
For f(x, y) = x² + 2xy + y²:

- ∂f/∂x = 2x + 2y (treat y as constant)
- ∂f/∂y = 2x + 2y (treat x as constant)

**Higher-Order Derivatives:**

- ∂²f/∂x² = second partial derivative (curvature)
- ∂²f/∂x∂y = mixed partial derivative
- Hessian matrix: contains all second-order partial derivatives

### 2.3 Gradients

**Definition:** The gradient ∇f is a vector of all partial derivatives:

**∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]**

**Key Properties:**

1. **Direction:** Points in the direction of steepest ascent
2. **Magnitude:** Indicates how steep the ascent is
3. **Perpendicular to level curves:** Always perpendicular to contour lines of constant f

**Gradient Descent:**

- To minimize f, move in the **opposite** direction of the gradient
- Update rule: x_new = x_old - α·∇f(x_old)
- α is the learning rate (step size)

**Directional Derivative:**

- Rate of change in direction of unit vector u
- D_u f = ∇f · u
- Maximum when u is parallel to ∇f

**Applications:**

- Optimization algorithms (gradient descent, Adam, etc.)
- Backpropagation in deep learning
- Finding critical points (where ∇f = 0)

### 2.4 Taylor Series

**Definition:** Approximation of a function using polynomials:

**f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...**

**First-order (Linear) Approximation:**

- f(x) ≈ f(a) + ∇f(a)·(x-a)
- Used in gradient descent

**Second-order (Quadratic) Approximation:**

- f(x) ≈ f(a) + ∇f(a)·(x-a) + ½(x-a)ᵀH(a)(x-a)
- H is the Hessian matrix
- Used in Newton's method

---

## 3. Probability and Statistics Fundamentals

Understanding uncertainty is crucial for machine learning, especially for probabilistic models and evaluation metrics.

### 3.1 Probability Basics

**Sample Space (S):** Set of all possible outcomes

**Event:** A subset of the sample space

**Probability Axioms:**

1. 0 ≤ P(A) ≤ 1 for any event A
2. P(S) = 1 (something must happen)
3. For mutually exclusive events: P(A ∪ B) = P(A) + P(B)

**Key Probability Rules:**

1. **Addition Rule:**

   - P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

2. **Multiplication Rule (Independent Events):**

   - P(A ∩ B) = P(A) × P(B)

3. **Conditional Probability:**

   - P(A|B) = P(A ∩ B) / P(B)
   - Read as "probability of A given B"

4. **Bayes' Theorem:**
   - P(A|B) = P(B|A) × P(A) / P(B)
   - Fundamental for Bayesian inference and machine learning

**Law of Total Probability:**

- P(B) = Σᵢ P(B|Aᵢ) × P(Aᵢ)

### 3.2 Random Variables

**Definition:** A random variable is a function that assigns a numerical value to each outcome in a sample space.

**Types:**

1. **Discrete:** Takes countable values (e.g., coin flips, dice rolls)
2. **Continuous:** Takes any value in a range (e.g., height, temperature)

**Probability Mass Function (PMF):** For discrete variables

- P(X = x) = probability that X equals x
- Σ P(X = x) = 1 (over all possible x)

**Probability Density Function (PDF):** For continuous variables

- f(x) ≥ 0 for all x
- ∫ f(x)dx = 1 (over entire range)
- P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx

**Cumulative Distribution Function (CDF):**

- F(x) = P(X ≤ x)
- Monotonically increasing
- lim(x→-∞) F(x) = 0, lim(x→∞) F(x) = 1

### 3.3 Common Probability Distributions

**1. Normal (Gaussian) Distribution:**

- PDF: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
- Parameters: μ (mean), σ² (variance)
- Notation: N(μ, σ²)
- 68-95-99.7 Rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ
- Central Limit Theorem: Sum of many random variables → normal

**2. Binomial Distribution:**

- Models number of successes in n independent trials
- P(X = k) = C(n,k) pᵏ (1-p)ⁿ⁻ᵏ
- Parameters: n (trials), p (success probability)
- Mean: np, Variance: np(1-p)

**3. Uniform Distribution:**

- All outcomes equally likely
- PDF: f(x) = 1/(b-a) for x ∈ [a,b]
- Mean: (a+b)/2

**4. Exponential Distribution:**

- Models time between events
- PDF: f(x) = λe⁻ᵏˣ for x ≥ 0
- Mean: 1/λ, Variance: 1/λ²

### 3.4 Expected Value and Variance

**Expected Value (Mean):**

- E[X] = μ = Σ x·P(X=x) for discrete
- E[X] = ∫ x·f(x)dx for continuous
- Represents the "average" or "center" of the distribution

**Properties:**

- E[aX + b] = aE[X] + b (linearity)
- E[X + Y] = E[X] + E[Y]

**Variance:**

- Var(X) = σ² = E[(X - μ)²] = E[X²] - (E[X])²
- Measures spread/dispersion around the mean

**Standard Deviation:**

- σ = √Var(X)
- Same units as the original variable

**Properties:**

- Var(aX + b) = a²Var(X)
- For independent X, Y: Var(X + Y) = Var(X) + Var(Y)

### 3.5 Covariance and Correlation

**Covariance:**

- Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
- Measures how two variables change together
- Positive: tend to increase together
- Negative: one increases when other decreases
- Zero: no linear relationship

**Correlation Coefficient:**

- ρ(X, Y) = Cov(X, Y) / (σₓ · σᵧ)
- Normalized covariance: -1 ≤ ρ ≤ 1
- ρ = 1: perfect positive linear relationship
- ρ = -1: perfect negative linear relationship
- ρ = 0: no linear relationship

**Important:** Correlation ≠ Causation!

### 3.6 Statistical Measures

**Central Tendency:**

1. **Mean:** Average value = Σxᵢ / n
2. **Median:** Middle value when sorted
3. **Mode:** Most frequent value

**Spread:**

1. **Range:** Maximum - Minimum
2. **Interquartile Range (IQR):** Q₃ - Q₁
3. **Variance:** Average squared deviation from mean
4. **Standard Deviation:** Square root of variance

**Shape:**

1. **Skewness:** Measure of asymmetry
   - Positive skew: tail on right
   - Negative skew: tail on left
2. **Kurtosis:** Measure of "tailedness"
   - High kurtosis: heavy tails, more outliers

**Percentiles/Quantiles:**

- Qₖ: Value below which k% of data falls
- Q₁ (25th), Q₂ (50th = median), Q₃ (75th)

---

## 4. Gradient Descent: Theory

**Gradient Descent** is the fundamental optimization algorithm for training machine learning models.

### 4.1 The Optimization Problem

**Goal:** Find x\* that minimizes f(x)

- x\* = arg min f(x)

**Approach:** Start at some point and iteratively move toward the minimum

### 4.2 Gradient Descent Algorithm

**Core Idea:** Move in the direction opposite to the gradient (steepest descent)

**Update Rule:**
**xₜ₊₁ = xₜ - α·∇f(xₜ)**

Where:

- xₜ = current position at iteration t
- α = learning rate (step size)
- ∇f(xₜ) = gradient at current position
- xₜ₊₁ = new position

**Algorithm Steps:**

1. Initialize x₀ (starting point)
2. Compute gradient ∇f(xₜ)
3. Update: xₜ₊₁ = xₜ - α·∇f(xₜ)
4. Repeat until convergence

### 4.3 Convergence Criteria

**When to stop:**

1. **Gradient magnitude:** |∇f(x)| < ε (very small)
2. **Function change:** |f(xₜ₊₁) - f(xₜ)| < ε
3. **Parameter change:** |xₜ₊₁ - xₜ| < ε
4. **Maximum iterations:** Fixed number of steps

### 4.4 Learning Rate (α)

**Critical Hyperparameter:**

1. **Too Small (α → 0):**

   - ✅ Stable convergence
   - ❌ Very slow, many iterations needed
   - ❌ May get stuck in plateaus

2. **Too Large (α → ∞):**

   - ❌ Unstable, may diverge
   - ❌ Oscillates around minimum
   - ❌ May overshoot optimal point

3. **Optimal (Goldilocks α):**
   - ✅ Fast convergence
   - ✅ Stable updates
   - ✅ Reaches minimum efficiently

**Learning Rate Schedules:**

- **Constant:** α stays same
- **Time-based decay:** α = α₀ / (1 + kt)
- **Step decay:** Reduce α every k iterations
- **Exponential decay:** α = α₀ · e⁻ᵏᵗ
- **Adaptive:** Different α for each parameter (Adam, RMSprop)

### 4.5 Types of Gradient Descent

**1. Batch Gradient Descent:**

- Uses entire dataset to compute gradient
- ∇f = (1/N) Σᵢ ∇L(xᵢ)
- ✅ Accurate gradient
- ❌ Slow for large datasets
- ❌ High memory requirements

**2. Stochastic Gradient Descent (SGD):**

- Uses one random sample at a time
- ∇f ≈ ∇L(xᵢ) for random i
- ✅ Fast updates
- ✅ Can escape local minima
- ❌ Noisy, high variance
- ❌ May not converge exactly

**3. Mini-batch Gradient Descent:**

- Uses small batches (e.g., 32, 64, 128 samples)
- ∇f ≈ (1/B) Σᵢ₌₁ᴮ ∇L(xᵢ)
- ✅ Balance between accuracy and speed
- ✅ Efficient GPU utilization
- ✅ Most commonly used in practice

### 4.6 Challenges and Solutions

**Challenge 1: Local Minima**

- Non-convex functions have multiple local minima
- Solutions: Random restarts, momentum, simulated annealing

**Challenge 2: Saddle Points**

- Points where gradient is zero but not a minimum
- Solutions: Second-order methods, momentum

**Challenge 3: Plateaus**

- Flat regions with very small gradients
- Solutions: Adaptive learning rates, patience

**Challenge 4: Ill-conditioned Problems**

- Different dimensions have different scales
- Solutions: Feature normalization, preconditioning

### 4.7 Advanced Optimization Algorithms

**Momentum:**

- vₜ₊₁ = β·vₜ + ∇f(xₜ)
- xₜ₊₁ = xₜ - α·vₜ₊₁
- Accumulates velocity, smooths updates

**Nesterov Accelerated Gradient:**

- "Look ahead" before computing gradient
- Often converges faster than standard momentum

**AdaGrad:**

- Adapts learning rate per parameter
- Suitable for sparse data

**RMSprop:**

- Uses moving average of squared gradients
- Works well for non-stationary problems

**Adam (Adaptive Moment Estimation):**

- Combines momentum and RMSprop
- Most popular in deep learning
- Automatically adapts learning rates

---

## 5. Convexity and Optimization Landscape

### 5.1 Convex Functions

**Definition:** f is convex if for any x₁, x₂ and λ ∈ [0,1]:
**f(λx₁ + (1-λ)x₂) ≤ λf(x₁) + (1-λ)f(x₂)**

**Properties:**

- Any local minimum is a global minimum
- Gradient descent guaranteed to find global minimum
- Examples: Linear regression, logistic regression

**Convex Optimization:**

- Well-studied, efficient algorithms
- Strong convergence guarantees
- Practical and theoretical importance

### 5.2 Non-Convex Optimization

**Deep Learning Challenge:**

- Neural networks have non-convex loss functions
- Multiple local minima and saddle points
- No guarantee of finding global minimum

**Why It Works Anyway:**

- Many local minima are "good enough"
- Over-parameterization helps (wide networks)
- Stochastic gradient descent provides regularization
- Modern architectures designed for trainability

---

## 6. Exercise: Gradient Descent Implementation

**Objective:** Implement gradient descent from scratch to minimize a function.

**Problem:** Minimize f(x, y) = x² + y²

**Steps:**

1. Define the objective function: f(x, y) = x² + y²
2. Compute gradient: ∇f = [2x, 2y]
3. Initialize starting point (e.g., x₀ = [5, 5])
4. Set learning rate (e.g., α = 0.1)
5. Iteratively update: [x, y]ₜ₊₁ = [x, y]ₜ - α·[2x, 2y]ₜ
6. Track progress until convergence
7. Verify: Should converge to [0, 0] where f(0,0) = 0

**Expected Behavior:**

- Function value should decrease monotonically
- Position should spiral toward origin
- Convergence depends on learning rate choice

**Analysis:**

- Try different learning rates: 0.01, 0.1, 0.5, 0.9
- Observe convergence speed and stability
- Plot optimization path on contour map

---

## 7. Best Practices for Mathematical ML

### 7.1 Numerical Stability

**Common Issues:**

1. **Overflow:** Numbers become too large
2. **Underflow:** Numbers become too small
3. **Loss of precision:** Subtraction of similar numbers

**Solutions:**

- Log-space computations for small probabilities
- Numerical tricks (log-sum-exp)
- Careful ordering of operations
- Use of numerically stable formulas

### 7.2 Computational Efficiency

**Vectorization:**

- Replace loops with matrix operations
- 10-100x speedup typical
- Leverages optimized linear algebra libraries

**Broadcasting:**

- Implicit expansion of dimensions
- Avoids explicit loops and copies

**Memory Management:**

- In-place operations when possible
- Clear intermediate results
- Batch processing for large datasets

### 7.3 Debugging Mathematical Code

**Verification Strategies:**

1. **Gradient Checking:** Compare analytical vs numerical gradients
2. **Dimensionality Checking:** Verify matrix shapes
3. **Simple Test Cases:** Known solutions
4. **Visualization:** Plot intermediate results
5. **Unit Tests:** Test individual components

---

## 8. Next Steps

Congratulations! You've built a solid mathematical foundation for machine learning. You now understand:

- **Linear Algebra:** Vectors, matrices, eigenvalues, transformations
- **Calculus:** Derivatives, gradients, optimization theory
- **Probability & Statistics:** Distributions, correlation, statistical measures
- **Gradient Descent:** The core optimization algorithm powering ML

**Coming Up Next:**

- Day 8-10: Introduction to Machine Learning
- Scikit-learn fundamentals
- Classification and regression models
- Model evaluation and validation

---

## Additional Resources

### Linear Algebra

- **3Blue1Brown: Essence of Linear Algebra** - Visual, intuitive explanations
- **MIT OpenCourseWare: Linear Algebra** - Rigorous treatment by Gilbert Strang
- **Khan Academy: Linear Algebra** - Step-by-step tutorials

### Calculus

- **Khan Academy: Multivariable Calculus** - Comprehensive coverage
- **3Blue1Brown: Essence of Calculus** - Beautiful visualizations
- **MIT OpenCourseWare: Multivariable Calculus** - Advanced topics

### Probability & Statistics

- **Seeing Theory** - Interactive visual introduction to probability
- **StatQuest with Josh Starmer** - Fun, clear explanations
- **Khan Academy: Statistics and Probability** - Complete course

### Optimization

- **Convex Optimization** by Boyd & Vandenberghe - Free online textbook
- **Numerical Optimization** by Nocedal & Wright - Advanced reference

### Books

- **Mathematics for Machine Learning** by Deisenroth, Faisal, and Ong
  - Free PDF available online
  - Covers all topics needed for ML
- **Deep Learning** by Goodfellow, Bengio, and Courville

  - Chapters 2-4: Mathematical foundations
  - Free online version available

- **Pattern Recognition and Machine Learning** by Bishop
  - Comprehensive probabilistic perspective

### Practice

- **Brilliant.org** - Interactive problem-solving
- **Khan Academy** - Structured practice problems
- **MIT OCW Problem Sets** - Challenging exercises

---

**Keep Calculating! 📐✨**
