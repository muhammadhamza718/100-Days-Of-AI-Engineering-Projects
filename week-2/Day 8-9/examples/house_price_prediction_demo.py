"""
House Price Prediction Demo

This script demonstrates the complete implementation of regression models
for house price prediction using the from-scratch implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data.housing_data import load_sample_housing_data, generate_nonlinear_housing_data
from src.regression.linear_regression import LinearRegressor
from src.regression.polynomial_regression import PolynomialRegressor
from src.regression.regularization.ridge_regression import RidgeRegressor
from src.regression.regularization.lasso_regression import LassoRegressor
from src.integration.model_comparison import ModelComparisonFramework, run_model_comparison
from src.preprocessing.feature_scaling import standardize_features
from src.metrics.mse import calculate_mse, calculate_rmse, calculate_mae
from src.metrics.r_squared import calculate_r_squared
from src.utils.visualization import (plot_cost_convergence, plot_predictions_vs_actual,
                                   plot_residuals, plot_feature_importance)


def demo_linear_regression():
    """Demonstrate Linear Regression implementation."""
    print("=" * 60)
    print("LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)

    # Load sample housing data
    X, y = load_sample_housing_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    from src.data.housing_data import split_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, train_mean, train_std = standardize_features(X_train)
    X_test_scaled = (X_test - train_mean) / train_std

    # Create and train Linear Regression model
    model = LinearRegressor(learning_rate=0.01, max_iterations=1000)
    print("\nTraining Linear Regression model...")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    train_mse = calculate_mse(y_train, y_pred_train)
    test_mse = calculate_mse(y_test, y_pred_test)
    train_rmse = calculate_rmse(y_train, y_pred_train)
    test_rmse = calculate_rmse(y_test, y_pred_test)
    train_mae = calculate_mae(y_train, y_pred_train)
    test_mae = calculate_mae(y_test, y_pred_test)
    train_r2 = calculate_r_squared(y_train, y_pred_train)
    test_r2 = calculate_r_squared(y_test, y_pred_test)

    print(f"\nTraining Metrics:")
    print(f"  MSE: {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R²: {train_r2:.4f}")

    print(f"\nTesting Metrics:")
    print(f"  MSE: {test_mse:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R²: {test_r2:.4f}")

    # Show model parameters
    params = model.get_parameters()
    print(f"\nLearned Parameters (theta):")
    print(f"  Intercept: {params[0]:.2f}")
    print(f"  Coefficients: {params[1:]}")

    # Plot cost convergence
    if model.get_cost_history():
        plot_cost_convergence({"Linear Regression": model.get_cost_history()},
                             title="Linear Regression Cost Convergence")

    # Plot predictions vs actual
    plot_predictions_vs_actual(y_test, y_pred_test, "Linear Regression",
                              title="Linear Regression: Predictions vs Actual")

    # Plot residuals
    plot_residuals(y_test, y_pred_test, "Linear Regression")

    # Plot feature importance
    plot_feature_importance(params, ['Intercept', 'SqFt', 'Bedrooms', 'Age'],
                          "Linear Regression")


def demo_polynomial_regression():
    """Demonstrate Polynomial Regression implementation."""
    print("\n" + "=" * 60)
    print("POLYNOMIAL REGRESSION DEMONSTRATION")
    print("=" * 60)

    # Generate nonlinear data (using only one feature for visualization)
    X, y = generate_nonlinear_housing_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    from src.data.housing_data import split_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, train_mean, train_std = standardize_features(X_train)
    X_test_scaled = (X_test - train_mean) / train_std

    # Create and train Polynomial Regression model (degree 2)
    model = PolynomialRegressor(degree=2, learning_rate=0.001, max_iterations=2000)
    print("\nTraining Polynomial Regression model (degree=2)...")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    train_r2 = calculate_r_squared(y_train, y_pred_train)
    test_r2 = calculate_r_squared(y_test, y_pred_test)

    print(f"\nTraining R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")

    # Show model parameters
    params = model.get_parameters()
    print(f"\nLearned Parameters (theta) - first 10:")
    print(f"  {params[:10]}{'...' if len(params) > 10 else ''}")

    # Plot cost convergence
    if model.get_cost_history():
        plot_cost_convergence({"Polynomial Regression": model.get_cost_history()},
                             title="Polynomial Regression Cost Convergence")

    # Plot predictions vs actual
    plot_predictions_vs_actual(y_test, y_pred_test, "Polynomial Regression",
                              title="Polynomial Regression: Predictions vs Actual")


def demo_regularization():
    """Demonstrate Ridge and Lasso Regression implementations."""
    print("\n" + "=" * 60)
    print("REGULARIZATION DEMONSTRATION (Ridge & Lasso)")
    print("=" * 60)

    # Load sample housing data
    X, y = load_sample_housing_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    from src.data.housing_data import split_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, train_mean, train_std = standardize_features(X_train)
    X_test_scaled = (X_test - train_mean) / train_std

    # Train Ridge Regression
    print("\nTraining Ridge Regression model...")
    ridge_model = RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
    ridge_model.fit(X_train_scaled, y_train)

    # Train Lasso Regression
    print("Training Lasso Regression model...")
    lasso_model = LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
    lasso_model.fit(X_train_scaled, y_train)

    # Make predictions
    ridge_pred = ridge_model.predict(X_test_scaled)
    lasso_pred = lasso_model.predict(X_test_scaled)

    # Calculate metrics
    ridge_r2 = calculate_r_squared(y_test, ridge_pred)
    lasso_r2 = calculate_r_squared(y_test, lasso_pred)

    print(f"\nRidge Regression R²: {ridge_r2:.4f}")
    print(f"Lasso Regression R²: {lasso_r2:.4f}")

    # Show parameters
    ridge_params = ridge_model.get_parameters()
    lasso_params = lasso_model.get_parameters()

    print(f"\nRidge Parameters (first 5): {ridge_params[:5]}")
    print(f"Lasso Parameters (first 5): {lasso_params[:5]}")

    # Show sparsity for Lasso
    lasso_sparsity = lasso_model.get_coefficient_sparsity()
    print(f"\nLasso Coefficient Sparsity: {lasso_sparsity:.2%}")


def demo_model_comparison():
    """Demonstrate model comparison framework."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)

    # Load sample housing data
    X, y = load_sample_housing_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Run model comparison
    framework = run_model_comparison(X, y, test_size=0.2, random_state=42,
                                   degree_range=(1, 3))


def main():
    """Run all demonstrations."""
    print("HOUSE PRICE PREDICTION DEMONSTRATION")
    print("Implementing regression models from scratch")
    print("Using mathematical formulas and gradient descent optimization")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run demonstrations
    demo_linear_regression()
    demo_polynomial_regression()
    demo_regularization()
    demo_model_comparison()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("All models implemented from scratch without using scikit-learn")
    print("for core algorithm logic as per the zero-library mandate.")


if __name__ == "__main__":
    main()