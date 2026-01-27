# API Contracts: House Price Prediction

## Regression Model Interface

### BaseRegressor Abstract Class

#### Methods

**fit(X, y)**
- **Input**:
  - `X`: array-like, shape (n_samples, n_features) - Training data features
  - `y`: array-like, shape (n_samples,) - Target values
- **Output**: Self (trained model)
- **Side Effects**: Updates internal model parameters
- **Errors**: ValueError if X and y have inconsistent shapes

**predict(X)**
- **Input**:
  - `X`: array-like, shape (n_samples, n_features) - Input data for prediction
- **Output**: array, shape (n_samples,) - Predicted target values
- **Side Effects**: None
- **Errors**: RuntimeError if called before fitting

**score(X, y)**
- **Input**:
  - `X`: array-like, shape (n_samples, n_features) - Input data
  - `y`: array-like, shape (n_samples,) - True target values
- **Output**: float - R² coefficient of determination
- **Side Effects**: None
- **Errors**: RuntimeError if called before fitting

### LinearRegressor Class

Extends: BaseRegressor

**Constructor Parameters**:
- `learning_rate`: float, default=0.01 - Step size for gradient descent
- `max_iterations`: int, default=1000 - Maximum number of iterations
- `tolerance`: float, default=1e-6 - Convergence threshold

### PolynomialRegressor Class

Extends: LinearRegressor

**Constructor Parameters**:
- `degree`: int, default=2 - Degree of polynomial features
- `learning_rate`: float, default=0.01 - Step size for gradient descent
- `max_iterations`: int, default=1000 - Maximum number of iterations
- `tolerance`: float, default=1e-6 - Convergence threshold

### Regularized Regressors

#### RidgeRegressor Class

Extends: LinearRegressor

**Constructor Parameters**:
- `lambda_reg`: float, default=0.01 - Regularization strength
- `learning_rate`: float, default=0.01 - Step size for gradient descent
- `max_iterations`: int, default=1000 - Maximum number of iterations
- `tolerance`: float, default=1e-6 - Convergence threshold

#### LassoRegressor Class

Extends: LinearRegressor

**Constructor Parameters**:
- `lambda_reg`: float, default=0.01 - Regularization strength
- `learning_rate`: float, default=0.01 - Step size for gradient descent
- `max_iterations`: int, default=1000 - Maximum number of iterations
- `tolerance`: float, default=1e-6 - Convergence threshold

## Preprocessing Functions

### normalize_features(X)
- **Input**: `X`: array-like, shape (n_samples, n_features)
- **Output**: Tuple (X_normalized, mean_vector, std_vector)
- **Description**: Normalizes features to have mean=0 and std=1

### generate_polynomial_features(X, degree)
- **Input**:
  - `X`: array-like, shape (n_samples, n_features)
  - `degree`: int - Degree of polynomial expansion
- **Output**: array-like, shape (n_samples, n_output_features)
- **Description**: Generates polynomial and interaction features

## Evaluation Functions

### calculate_mse(y_true, y_pred)
- **Input**:
  - `y_true`: array-like, true target values
  - `y_pred`: array-like, predicted target values
- **Output**: float - Mean squared error

### calculate_r_squared(y_true, y_pred)
- **Input**:
  - `y_true`: array-like, true target values
  - `y_pred`: array-like, predicted target values
- **Output**: float - R² coefficient of determination

### calculate_rmse(y_true, y_pred)
- **Input**:
  - `y_true`: array-like, true target values
  - `y_pred`: array-like, predicted target values
- **Output**: float - Root mean squared error

### calculate_mae(y_true, y_pred)
- **Input**:
  - `y_true`: array-like, true target values
  - `y_pred`: array-like, predicted target values
- **Output**: float - Mean absolute error