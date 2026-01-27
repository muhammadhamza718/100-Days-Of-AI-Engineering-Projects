# Data Model: House Price Prediction

## Core Entities

### HousingData
**Description**: Represents a collection of housing properties with features and target values
**Fields**:
- `square_footage`: float - Size of the house in square feet
- `bedrooms`: int - Number of bedrooms in the house
- `age`: float - Age of the house in years
- `price`: float - Target variable (house price)

**Validation Rules**:
- square_footage > 0
- bedrooms >= 0
- age >= 0
- price > 0

### RegressionParameters
**Description**: Stores the learned parameters of a regression model
**Fields**:
- `theta`: array[float] - Model parameters (weights) including intercept
- `intercept`: float - Bias term (θ₀)
- `cost_history`: array[float] - Historical cost values during training

### ModelConfiguration
**Description**: Configuration settings for regression models
**Fields**:
- `learning_rate`: float - Step size for gradient descent (α)
- `max_iterations`: int - Maximum number of training iterations
- `lambda_reg`: float - Regularization strength (λ)
- `degree`: int - Polynomial degree (for polynomial regression)

### TrainingResult
**Description**: Result of a model training operation
**Fields**:
- `model_parameters`: RegressionParameters - Learned model parameters
- `final_cost`: float - Final cost after training
- `iterations_completed`: int - Number of iterations performed
- `converged`: bool - Whether the model converged

### EvaluationMetrics
**Description**: Performance metrics for model evaluation
**Fields**:
- `mse`: float - Mean Squared Error
- `rmse`: float - Root Mean Squared Error
- `mae`: float - Mean Absolute Error
- `r_squared`: float - Coefficient of determination
- `adjusted_r_squared`: float - Adjusted R-squared (accounts for features)

## Relationships

```
HousingData -(used for training)-> RegressionModel
ModelConfiguration -(configures)-> RegressionModel
RegressionParameters -(result of training)-> RegressionModel
TrainingResult -(contains)-> RegressionParameters
EvaluationMetrics -(evaluates)-> TrainingResult
```

## State Transitions

### RegressionModel States
1. **Initialized**: Model created with configuration
2. **Training**: Model parameters being updated via gradient descent
3. **Trained**: Model has learned parameters and can make predictions
4. **Evaluated**: Model performance metrics computed

### Training Process
```
[Initialized] --(fit(X, y))--> [Training] --(convergence/max iterations)--> [Trained] --(evaluate(X_test, y_test))--> [Evaluated]
```