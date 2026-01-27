"""
Visualization utilities for regression models.

This module provides utilities for visualizing regression model performance,
including cost function convergence, predictions vs actual values, and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from src.regression.linear_regression import LinearRegressor
from src.regression.polynomial_regression import PolynomialRegressor
from src.regression.regularization.ridge_regression import RidgeRegressor
from src.regression.regularization.lasso_regression import LassoRegressor


def plot_cost_convergence(cost_histories: Dict[str, List[float]], title: str = "Cost Function Convergence") -> None:
    """
    Plot the convergence of cost functions for different models.

    Args:
        cost_histories (Dict[str, List[float]]): Dictionary mapping model names to their cost histories
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 8))

    for model_name, cost_history in cost_histories.items():
        plt.plot(cost_history, label=model_name, linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model",
                              title: str = "Predictions vs Actual Values") -> None:
    """
    Plot predicted values vs actual values to visualize model performance.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)

    # Add perfect prediction line (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.title(f"{title} - {model_name}", fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model",
                   title: str = "Residual Plot") -> None:
    """
    Plot residuals (errors) to check for patterns.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model
        title (str): Title for the plot
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 8))

    # Plot residuals vs predicted values
    plt.scatter(y_pred, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')

    plt.title(f"{title} - {model_name}", fontsize=16)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(coefficients: np.ndarray, feature_names: Optional[List[str]] = None,
                          model_name: str = "Model", title: str = "Feature Importance") -> None:
    """
    Plot the magnitude of model coefficients to assess feature importance.

    Args:
        coefficients (np.ndarray): Model coefficients (excluding bias term)
        feature_names (Optional[List[str]]): Names of the features
        model_name (str): Name of the model
        title (str): Title for the plot
    """
    # Exclude bias term (first coefficient)
    coefs = coefficients[1:] if len(coefficients) > 1 else coefficients

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(coefs))]
    elif len(feature_names) != len(coefs):
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match "
                         f"number of coefficients ({len(coefs)})")

    plt.figure(figsize=(10, 6))

    # Create bar chart of coefficient magnitudes
    bars = plt.bar(feature_names, np.abs(coefs), color='skyblue', edgecolor='navy', linewidth=1)

    # Color bars based on positive/negative coefficients
    for i, coef in enumerate(coefs):
        bars[i].set_color('lightcoral' if coef < 0 else 'skyblue')

    plt.title(f"{title} - {model_name}", fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Magnitude', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_regularization_path(models: List[object], regularization_param: str = 'lambda_reg',
                           param_values: List[float] = None, X: np.ndarray = None, y: np.ndarray = None,
                           title: str = "Regularization Path") -> None:
    """
    Plot how model coefficients change with different regularization strengths.

    Args:
        models (List[object]): List of trained models with different regularization strengths
        regularization_param (str): Name of the regularization parameter
        param_values (List[float]): Values of the regularization parameter used
        X (np.ndarray): Feature matrix (needed if models need to be re-trained)
        y (np.ndarray): Target values (needed if models need to be re-trained)
        title (str): Title for the plot
    """
    if param_values is None:
        # Try to extract parameter values from models
        param_values = []
        for model in models:
            if hasattr(model, regularization_param):
                param_values.append(getattr(model, regularization_param))
            else:
                param_values.append(0.0)  # Default value if attribute doesn't exist

    if len(param_values) != len(models):
        raise ValueError("Number of parameter values must match number of models")

    plt.figure(figsize=(12, 8))

    # Collect coefficients for each regularization strength
    all_coefs = []
    for model in models:
        if hasattr(model, 'get_parameters') and model.get_parameters() is not None:
            # Exclude bias term (first coefficient)
            coefs = model.get_parameters()[1:]
            all_coefs.append(coefs)
        else:
            all_coefs.append(np.array([]))

    # Find the maximum number of coefficients across all models
    max_coefs = max([len(coefs) for coefs in all_coefs]) if all_coefs else 0

    if max_coefs > 0:
        # Pad coefficient arrays to the same length
        padded_coefs = []
        for coefs in all_coefs:
            padded = np.pad(coefs, (0, max_coefs - len(coefs)), mode='constant', constant_values=0)
            padded_coefs.append(padded)

        # Convert to numpy array for easier manipulation
        coefs_array = np.array(padded_coefs)

        # Plot each coefficient as a function of regularization parameter
        for i in range(max_coefs):
            plt.plot(param_values, coefs_array[:, i], marker='o', label=f'Coefficient {i+1}', linewidth=2)

        plt.title(f"{title}", fontsize=16)
        plt.xlabel(regularization_param.capitalize(), fontsize=14)
        plt.ylabel('Coefficient Value', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Use log scale for regularization parameter
        plt.tight_layout()
        plt.show()


def plot_polynomial_fit(X: np.ndarray, y: np.ndarray, model: PolynomialRegressor,
                       title: str = "Polynomial Regression Fit") -> None:
    """
    Plot polynomial regression fit against data (for 1D feature case).

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        model (PolynomialRegressor): Trained polynomial regression model
        title (str): Title for the plot
    """
    if X.shape[1] != 1:
        print("Polynomial fit plot is only available for 1D features")
        return

    plt.figure(figsize=(10, 8))

    # Sort the data for smooth curve plotting
    sorted_indices = np.argsort(X.flatten())
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    # Plot original data
    plt.scatter(X_sorted, y_sorted, alpha=0.6, label='Data Points', s=50)

    # Generate smooth curve for prediction
    X_smooth = np.linspace(X_sorted.min(), X_sorted.max(), 300).reshape(-1, 1)
    y_pred_smooth = model.predict(X_smooth)

    plt.plot(X_smooth, y_pred_smooth, color='red', linewidth=2, label=f'Polynomial Fit (degree {model.degree})')

    plt.title(f"{title}", fontsize=16)
    plt.xlabel('Feature Value', fontsize=14)
    plt.ylabel('Target Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_complexity_vs_performance(train_scores: List[float], test_scores: List[float],
                                       complexity_values: List[float], complexity_label: str = "Complexity",
                                       title: str = "Model Complexity vs Performance") -> None:
    """
    Plot model performance vs complexity (e.g., polynomial degree, number of features).

    Args:
        train_scores (List[float]): Training scores for different complexity values
        test_scores (List[float]): Test scores for different complexity values
        complexity_values (List[float]): Complexity values (e.g., polynomial degrees)
        complexity_label (str): Label for the complexity axis
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 8))

    plt.plot(complexity_values, train_scores, marker='o', label='Training Score', linewidth=2)
    plt.plot(complexity_values, test_scores, marker='s', label='Test Score', linewidth=2)

    plt.title(f"{title}", fontsize=16)
    plt.xlabel(complexity_label, fontsize=14)
    plt.ylabel('Performance Score', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_validation_curve(train_scores: np.ndarray, val_scores: np.ndarray,
                         param_range: List[float], param_name: str,
                         title: str = "Validation Curve") -> None:
    """
    Plot validation curve showing training and validation scores for different parameter values.

    Args:
        train_scores (np.ndarray): Training scores for each parameter value
        val_scores (np.ndarray): Validation scores for each parameter value
        param_range (List[float]): Parameter values
        param_name (str): Name of the parameter
        title (str): Title for the plot
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 8))

    plt.plot(param_range, train_mean, label='Training Score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

    plt.plot(param_range, val_mean, label='Validation Score', color='red', marker='s')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.title(f"{title}", fontsize=16)
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_dashboard(model_results: Dict[str, Dict[str, float]], figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Create a dashboard showing multiple model performance metrics.

    Args:
        model_results (Dict[str, Dict[str, float]]): Results dictionary from model evaluation
        figsize (Tuple[int, int]): Figure size for the dashboard
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)

    # Extract metrics
    test_r2_scores = [model_results[name]['test_r2'] for name in model_names]
    test_rmse_scores = [model_results[name]['test_rmse'] for name in model_names]
    overfitting_indicators = [model_results[name]['overfitting_indicator'] for name in model_names]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Bar chart of test R² scores
    axes[0, 0].bar(model_names, test_r2_scores, color='lightblue', edgecolor='navy')
    axes[0, 0].set_title('Test R² Scores')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Bar chart of test RMSE scores
    axes[0, 1].bar(model_names, test_rmse_scores, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_title('Test RMSE Scores')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Bar chart of overfitting indicators
    colors = ['lightcoral' if x > 0.1 else 'lightyellow' if x > 0.05 else 'lightgreen'
              for x in overfitting_indicators]
    axes[1, 0].bar(model_names, overfitting_indicators, color=colors, edgecolor='black')
    axes[1, 0].set_title('Overfitting Indicators (Train R² - Test R²)')
    axes[1, 0].set_ylabel('Overfitting Indicator')
    axes[1, 0].axhline(y=0.05, color='gray', linestyle='--', alpha=0.7, label='Minor Overfitting Threshold')
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting Threshold')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Scatter plot of R² vs RMSE
    scatter = axes[1, 1].scatter(test_r2_scores, test_rmse_scores, s=100, c=range(n_models), cmap='viridis')
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name, (test_r2_scores[i], test_rmse_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Test R²')
    axes[1, 1].set_ylabel('Test RMSE')
    axes[1, 1].set_title('R² vs RMSE Comparison')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()