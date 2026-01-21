# Mathematics for Machine Learning Foundations

Implementation of mathematical foundations for ML from scratch, including linear algebra, calculus, probability, and optimization algorithms.

## Project Structure

- `linear_algebra/` - Vector and matrix operations, eigenvalues
- `calculus/` - Derivatives, gradients
- `probability/` - Distributions, statistics
- `optimization/` - Gradient descent and optimization algorithms
- `exercises/` - Practical implementations and exercises
- `tests/` - Unit and integration tests
- `outputs/plots/` - Generated visualizations

## Installation

```bash
pip install -e .
# or with uv
uv add numpy matplotlib scipy pytest
```

## Usage

### Running Gradient Descent Exercises

```bash
# Simple quadratic optimization
python exercises/gradient_descent_simple.py

# Advanced Rosenbrock function
python exercises/gradient_descent_advanced.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=. tests/
```

## Implementation Status

- [x] Project structure and dependencies
- [x] Gradient descent algorithm
- [ ] Linear algebra operations
- [ ] Calculus operations
- [ ] Probability distributions
- [x] Exercises implementations
- [x] Testing suite
- [x] Visualization tools