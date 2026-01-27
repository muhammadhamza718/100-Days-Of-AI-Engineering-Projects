# Quickstart: Mathematics for Machine Learning Foundations

## Setup

1. Install dependencies:
```bash
pip install -e .
# or with uv
uv add numpy matplotlib scipy pytest
```

2. Verify installation:
```bash
python -c "import numpy, matplotlib, scipy, pytest; print('Dependencies installed')"
```

## Running Gradient Descent Exercise

1. Run the simple quadratic optimization:
```bash
python exercises/gradient_descent_simple.py
```

2. Run the advanced Rosenbrock optimization:
```bash
python exercises/gradient_descent_advanced.py
```

3. Check generated visualizations in `outputs/plots/`

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Check test coverage:
```bash
pytest --cov=. tests/
```

## Key Components

- `optimization/gradient_descent.py`: Core gradient descent implementation
- `linear_algebra/`: Vector and matrix operations from scratch
- `calculus/`: Derivative and gradient calculations
- `probability/`: Probability distributions and statistics
- `exercises/`: Practical optimization examples
- `outputs/plots/`: Generated visualizations