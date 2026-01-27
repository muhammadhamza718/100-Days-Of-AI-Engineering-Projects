# Quickstart Guide: House Price Prediction

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd <repo-directory>
```

2. Install dependencies:
```bash
pip install numpy matplotlib pytest
```

## Basic Usage

### 1. Linear Regression Example
```python
from src.regression.linear_regression import LinearRegressor
import numpy as np

# Sample housing data (sqft, bedrooms, age)
X = np.array([[2100, 3, 10],
              [1600, 2, 15],
              [2400, 4, 5],
              [1400, 2, 20]])

# Prices in thousands
y = np.array([400, 330, 369, 232])

# Create and train model
model = LinearRegressor(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

### 2. Polynomial Regression Example
```python
from src.regression.polynomial_regression import PolynomialRegressor

# Create polynomial model (degree 2)
poly_model = PolynomialRegressor(degree=2, learning_rate=0.01, max_iterations=1000)
poly_model.fit(X, y)

# Make predictions
poly_predictions = poly_model.predict(X)
```

### 3. Regularized Regression Example
```python
from src.regression.regularization.ridge_regression import RidgeRegressor
from src.regression.regularization.lasso_regression import LassoRegressor

# Ridge Regression
ridge_model = RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
ridge_model.fit(X, y)

# Lasso Regression
lasso_model = LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
lasso_model.fit(X, y)
```

### 4. Model Evaluation
```python
from src.metrics.mse import calculate_mse
from src.metrics.r_squared import calculate_r_squared

# Calculate metrics
mse = calculate_mse(y_true=y, y_pred=predictions)
r2 = calculate_r_squared(y_true=y, y_pred=predictions)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
```

## Preprocessing Data

### Feature Scaling
```python
from src.preprocessing.feature_scaling import normalize_features

# Normalize features to have mean=0 and std=1
X_normalized = normalize_features(X)
```

### Polynomial Feature Generation
```python
from src.preprocessing.polynomial_features import generate_polynomial_features

# Generate polynomial features up to degree 2
X_poly = generate_polynomial_features(X, degree=2)
```

## Running Tests

```bash
pytest tests/unit/
pytest tests/integration/
```

## Training Pipeline Example

```python
from src.regression.linear_regression import LinearRegressor
from src.preprocessing.feature_scaling import normalize_features
from src.metrics.mse import calculate_mse

def train_housing_model(X_train, y_train, X_test, y_test):
    # Preprocess data
    X_train_norm = normalize_features(X_train)
    X_test_norm = normalize_features(X_test)  # Use same scaling parameters

    # Create and train model
    model = LinearRegressor(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train_norm, y_train)

    # Evaluate model
    train_pred = model.predict(X_train_norm)
    test_pred = model.predict(X_test_norm)

    train_mse = calculate_mse(y_train, train_pred)
    test_mse = calculate_mse(y_test, test_pred)

    return model, train_mse, test_mse
```