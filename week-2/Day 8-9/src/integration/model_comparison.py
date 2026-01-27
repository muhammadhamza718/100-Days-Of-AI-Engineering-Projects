"""
Model comparison framework for regression models.

This module provides utilities for training and comparing multiple regression models
including Linear, Polynomial, Ridge, and Lasso regressors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from src.regression.linear_regression import LinearRegressor
from src.regression.polynomial_regression import PolynomialRegressor
from src.regression.regularization.ridge_regression import RidgeRegressor
from src.regression.regularization.lasso_regression import LassoRegressor
from src.metrics.mse import calculate_mse
from src.metrics.r_squared import calculate_r_squared, calculate_adjusted_r_squared
from src.metrics.rmse_mae import calculate_rmse, calculate_mae
from src.data.housing_data import split_data
from src.preprocessing.feature_scaling import standardize_features, apply_standardization


class ModelComparisonFramework:
    """
    A framework for training and comparing different regression models.
    """

    def __init__(self):
        """Initialize the model comparison framework."""
        self.models = {}
        self.results = {}
        self.trained_models = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler_X = None
        self.scaler_y = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_state: Optional[int] = 42, scale_features: bool = True) -> 'ModelComparisonFramework':
        """
        Prepare data for model comparison by splitting and scaling.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            test_size (float): Proportion of data to use for testing
            random_state (Optional[int]): Random seed for reproducibility
            scale_features (bool): Whether to scale features

        Returns:
            ModelComparisonFramework: Self for chaining
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features if requested
        if scale_features:
            self.X_train, self.scaler_X_mean, self.scaler_X_std = standardize_features(self.X_train)
            self.X_test = apply_standardization(self.X_test, self.scaler_X_mean, self.scaler_X_std)

        return self

    def add_model(self, name: str, model) -> 'ModelComparisonFramework':
        """
        Add a model to the comparison framework.

        Args:
            name (str): Name of the model
            model: Model instance

        Returns:
            ModelComparisonFramework: Self for chaining
        """
        self.models[name] = model
        return self

    def train_all_models(self) -> 'ModelComparisonFramework':
        """
        Train all registered models.

        Returns:
            ModelComparisonFramework: Self for chaining
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            print(f"{name} trained successfully")

        return self

    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of evaluation results for each model
        """
        results = {}

        for name, model in self.trained_models.items():
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Calculate metrics
            train_mse = calculate_mse(self.y_train, y_pred_train)
            test_mse = calculate_mse(self.y_test, y_pred_test)

            train_rmse = calculate_rmse(self.y_train, y_pred_train)
            test_rmse = calculate_rmse(self.y_test, y_pred_test)

            train_mae = calculate_mae(self.y_train, y_pred_train)
            test_mae = calculate_mae(self.y_test, y_pred_test)

            train_r2 = calculate_r_squared(self.y_train, y_pred_train)
            test_r2 = calculate_r_squared(self.y_test, y_pred_test)

            # Calculate adjusted R² if model has information about number of features
            n_features = self.X_train.shape[1]
            if hasattr(model, 'degree'):  # For polynomial regression
                # Count the number of polynomial features
                from src.preprocessing.polynomial_features import count_polynomial_features
                n_features = count_polynomial_features(self.X_train.shape[1], model.degree)
            elif hasattr(model, 'original_n_features'):  # For polynomial regression
                from src.preprocessing.polynomial_features import count_polynomial_features
                n_features = count_polynomial_features(model.original_n_features, model.degree)

            train_adj_r2 = calculate_adjusted_r_squared(self.y_train, y_pred_train, n_features)
            test_adj_r2 = calculate_adjusted_r_squared(self.y_test, y_pred_test, n_features)

            # Calculate overfitting indicator (difference between train and test performance)
            overfitting_indicator = train_r2 - test_r2

            results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_adj_r2': train_adj_r2,
                'test_adj_r2': test_adj_r2,
                'overfitting_indicator': overfitting_indicator,
                'n_features': n_features
            }

        self.results = results
        return results

    def compare_models(self) -> str:
        """
        Generate a comparison report of all models.

        Returns:
            str: Formatted comparison report
        """
        if not self.results:
            raise RuntimeError("Models must be trained and evaluated before comparison")

        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"

        # Sort models by test R² score (descending)
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_r2'], reverse=True)

        for name, metrics in sorted_models:
            report += f"{name}:\n"
            report += f"  Test R²: {metrics['test_r2']:.4f}\n"
            report += f"  Test RMSE: {metrics['test_rmse']:.4f}\n"
            report += f"  Test MAE: {metrics['test_mae']:.4f}\n"
            report += f"  Overfitting Indicator: {metrics['overfitting_indicator']:.4f}\n"
            report += f"  Features: {metrics['n_features']}\n"
            report += "\n"

        # Identify best model based on test R²
        best_model_name = sorted_models[0][0]
        best_test_r2 = sorted_models[0][1]['test_r2']

        report += f"Best Model: {best_model_name} (Test R² = {best_test_r2:.4f})\n"

        return report

    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot a comparison of model performances.

        Args:
            figsize (Tuple[int, int]): Figure size for the plot
        """
        if not self.results:
            raise RuntimeError("Models must be trained and evaluated before plotting")

        # Prepare data for plotting
        model_names = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.results[name]['test_rmse'] for name in model_names]
        overfitting_indicators = [self.results[name]['overfitting_indicator'] for name in model_names]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Test R² scores
        axes[0, 0].bar(model_names, test_r2_scores)
        axes[0, 0].set_title('Test R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Test RMSE scores
        axes[0, 1].bar(model_names, test_rmse_scores)
        axes[0, 1].set_title('Test RMSE Scores')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Overfitting indicators
        colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in overfitting_indicators]
        axes[1, 0].bar(model_names, overfitting_indicators, color=colors)
        axes[1, 0].set_title('Overfitting Indicators (Train R² - Test R²)')
        axes[1, 0].set_ylabel('Overfitting Indicator')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: R² vs RMSE scatter
        axes[1, 1].scatter(test_r2_scores, test_rmse_scores, s=100)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (test_r2_scores[i], test_rmse_scores[i]),
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Test R²')
        axes[1, 1].set_ylabel('Test RMSE')
        axes[1, 1].set_title('R² vs RMSE Comparison')

        plt.tight_layout()
        plt.show()

    def get_best_model(self, metric: str = 'test_r2') -> Tuple[str, object]:
        """
        Get the best performing model based on a specific metric.

        Args:
            metric (str): Metric to use for comparison ('test_r2', 'test_rmse', etc.)

        Returns:
            Tuple[str, object]: Name and instance of the best model
        """
        if not self.results:
            raise RuntimeError("Models must be trained and evaluated first")

        best_model_name = max(self.results.keys(),
                             key=lambda x: self.results[x][metric])
        best_model = self.trained_models[best_model_name]

        return best_model_name, best_model


def create_default_models(X: np.ndarray, degree_range: Tuple[int, int] = (1, 3)) -> Dict[str, object]:
    """
    Create a set of default models for comparison.

    Args:
        X (np.ndarray): Feature matrix (used to determine number of features)
        degree_range (Tuple[int, int]): Range of polynomial degrees to test

    Returns:
        Dict[str, object]: Dictionary of model instances
    """
    n_features = X.shape[1]

    models = {
        'Linear Regression': LinearRegressor(learning_rate=0.01, max_iterations=1000),
        'Ridge Regression': RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000),
        'Lasso Regression': LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=1000)
    }

    # Add polynomial regressions for different degrees
    for degree in range(degree_range[0], degree_range[1] + 1):
        models[f'Polynomial Regression (deg={degree})'] = PolynomialRegressor(
            degree=degree,
            learning_rate=0.001,  # Lower learning rate for polynomial features
            max_iterations=2000   # More iterations for polynomial features
        )

    return models


def run_model_comparison(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                        random_state: Optional[int] = 42, scale_features: bool = True,
                        degree_range: Tuple[int, int] = (1, 3)) -> ModelComparisonFramework:
    """
    Run a complete model comparison workflow.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        test_size (float): Proportion of data to use for testing
        random_state (Optional[int]): Random seed for reproducibility
        scale_features (bool): Whether to scale features
        degree_range (Tuple[int, int]): Range of polynomial degrees to test

    Returns:
        ModelComparisonFramework: Framework with trained and evaluated models
    """
    # Create the comparison framework
    framework = ModelComparisonFramework()

    # Prepare data
    framework.prepare_data(X, y, test_size=test_size, random_state=random_state,
                          scale_features=scale_features)

    # Create and add models
    models = create_default_models(X, degree_range)
    for name, model in models.items():
        framework.add_model(name, model)

    # Train all models
    framework.train_all_models()

    # Evaluate all models
    framework.evaluate_all_models()

    # Print comparison report
    print(framework.compare_models())

    return framework