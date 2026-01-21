# API Contracts: Mathematics for Machine Learning Foundations

## Gradient Descent Function

**Endpoint**: `optimization.gradient_descent()`

**Signature**:
```python
def gradient_descent(
    objective_fn: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    start_point: np.ndarray,
    learning_rate: float,
    num_iterations: int
) -> Tuple[List[np.ndarray], List[float]]
```

**Parameters**:
- `objective_fn`: Function to minimize (Callable accepting ndarray, returning float)
- `gradient_fn`: Gradient of objective function (Callable accepting ndarray, returning ndarray)
- `start_point`: Initial parameter values (NumPy array)
- `learning_rate`: Step size for each iteration (float > 0)
- `num_iterations`: Number of iterations to perform (int >= 0)

**Returns**:
- Tuple of (path of parameter values, path of objective function values)
- Both lists have length (num_iterations + 1)

**Contract**:
- Precondition: objective_fn and gradient_fn must be compatible
- Postcondition: Returns sequence of optimization steps
- Error behavior: Raises appropriate exceptions for invalid inputs

## Mathematical Operations

**Linear Algebra Operations**:
- Vector addition, subtraction, scalar multiplication
- Matrix multiplication, transpose, determinant (from scratch)
- Dot product, norm calculations

**Calculus Operations**:
- Numerical differentiation (central difference method)
- Gradient computation for multivariable functions
- Partial derivative calculations

**Probability Operations**:
- Probability density function evaluations
- Random sampling from distributions
- Statistical moment calculations