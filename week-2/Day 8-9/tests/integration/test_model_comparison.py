"""
Integration tests for model comparison framework.

This module tests the integration of multiple regression models in the comparison framework.
"""

import numpy as np
import pytest
from src.integration.model_comparison import ModelComparisonFramework, create_default_models, run_model_comparison
from src.data.housing_data import load_sample_housing_data
from src.preprocessing.feature_scaling import standardize_features


def test_model_comparison_framework_initialization():
    """Test ModelComparisonFramework initialization."""
    framework = ModelComparisonFramework()

    assert framework.models == {}
    assert framework.results == {}
    assert framework.trained_models == {}
    assert framework.X_train is None
    assert framework.y_train is None
    assert framework.X_test is None
    assert framework.y_test is None


def test_model_comparison_framework_prepare_data():
    """Test data preparation in ModelComparisonFramework."""
    # Create simple data
    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    framework = ModelComparisonFramework()
    framework.prepare_data(X, y, test_size=0.2, random_state=42, scale_features=True)

    # Verify data has been split
    assert framework.X_train is not None
    assert framework.X_test is not None
    assert framework.y_train is not None
    assert framework.y_test is not None

    # Verify split proportions
    assert framework.X_train.shape[0] + framework.X_test.shape[0] == 100
    assert framework.y_train.shape[0] + framework.y_test.shape[0] == 100

    # Verify features are scaled
    assert np.allclose(np.mean(framework.X_train, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(framework.X_train, axis=0), 1, atol=1e-10)


def test_create_default_models():
    """Test creation of default models."""
    X = np.random.rand(50, 2)  # 2 features
    degree_range = (1, 2)

    models = create_default_models(X, degree_range)

    # Verify expected models are created
    expected_model_names = [
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Polynomial Regression (deg=1)',
        'Polynomial Regression (deg=2)'
    ]

    assert len(models) == len(expected_model_names)
    for name in expected_model_names:
        assert name in models


def test_run_model_comparison_integration():
    """Test the complete model comparison workflow."""
    # Use a smaller subset of the housing data for faster testing
    X_full, y_full = load_sample_housing_data()

    # Use only a subset for faster testing
    X = X_full[:50]
    y = y_full[:50]

    # Run model comparison
    framework = run_model_comparison(X, y, test_size=0.3, random_state=42,
                                   degree_range=(1, 2))

    # Verify that models were trained and evaluated
    assert len(framework.trained_models) > 0
    assert len(framework.results) > 0

    # Verify that results contain expected metrics
    for model_name, metrics in framework.results.items():
        expected_metrics = [
            'train_mse', 'test_mse', 'train_rmse', 'test_rmse',
            'train_mae', 'test_mae', 'train_r2', 'test_r2',
            'train_adj_r2', 'test_adj_r2', 'overfitting_indicator', 'n_features'
        ]

        for metric in expected_metrics:
            assert metric in metrics


def test_add_and_evaluate_single_model():
    """Test adding and evaluating a single model."""
    from src.regression.linear_regression import LinearRegressor

    # Create simple data
    np.random.seed(42)
    X = np.random.rand(60, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 1 + np.random.normal(0, 0.1, size=X.shape[0])

    framework = ModelComparisonFramework()
    framework.prepare_data(X, y, test_size=0.3, random_state=42, scale_features=True)

    # Add a single model
    model = LinearRegressor(learning_rate=0.01, max_iterations=500)
    framework.add_model('Test Linear Model', model)

    # Train the model
    framework.train_all_models()

    # Evaluate the model
    results = framework.evaluate_all_models()

    # Verify results contain the model
    assert 'Test Linear Model' in results

    # Verify expected metrics are present
    model_results = results['Test Linear Model']
    expected_metrics = [
        'train_mse', 'test_mse', 'train_rmse', 'test_rmse',
        'train_mae', 'test_mae', 'train_r2', 'test_r2',
        'train_adj_r2', 'test_adj_r2', 'overfitting_indicator', 'n_features'
    ]

    for metric in expected_metrics:
        assert metric in model_results


def test_model_comparison_report():
    """Test model comparison report generation."""
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(40, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 1 + np.random.normal(0, 0.1, size=X.shape[0])

    framework = ModelComparisonFramework()
    framework.prepare_data(X, y, test_size=0.3, random_state=42, scale_features=True)

    # Add a simple model
    from src.regression.linear_regression import LinearRegressor
    model = LinearRegressor(learning_rate=0.01, max_iterations=300)
    framework.add_model('Simple Linear Model', model)

    # Train and evaluate
    framework.train_all_models()
    framework.evaluate_all_models()

    # Generate comparison report
    report = framework.compare_models()

    # Verify report contains expected content
    assert 'Model Comparison Report' in report
    assert 'Simple Linear Model:' in report
    assert 'Best Model:' in report


def test_get_best_model():
    """Test getting the best performing model."""
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(40, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 1 + np.random.normal(0, 0.1, size=X.shape[0])

    framework = ModelComparisonFramework()
    framework.prepare_data(X, y, test_size=0.3, random_state=42, scale_features=True)

    # Add a simple model
    from src.regression.linear_regression import LinearRegressor
    model = LinearRegressor(learning_rate=0.01, max_iterations=300)
    framework.add_model('Simple Linear Model', model)

    # Train and evaluate
    framework.train_all_models()
    framework.evaluate_all_models()

    # Get best model based on test R²
    best_name, best_model = framework.get_best_model('test_r2')

    # Verify we get back the expected model
    assert best_name == 'Simple Linear Model'
    assert best_model is framework.trained_models['Simple Linear Model']


def test_model_comparison_with_real_data():
    """Test model comparison with real housing data."""
    # Use a small sample of housing data
    X_full, y_full = load_sample_housing_data()

    # Use only a subset for faster testing
    X = X_full[:30]
    y = y_full[:30]

    framework = ModelComparisonFramework()
    framework.prepare_data(X, y, test_size=0.3, random_state=42, scale_features=True)

    # Add multiple models
    from src.regression.linear_regression import LinearRegressor
    from src.regression.regularization.ridge_regression import RidgeRegressor
    from src.regression.regularization.lasso_regression import LassoRegressor

    framework.add_model('Linear Reg', LinearRegressor(learning_rate=0.01, max_iterations=500))
    framework.add_model('Ridge Reg', RidgeRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=500))
    framework.add_model('Lasso Reg', LassoRegressor(lambda_reg=0.1, learning_rate=0.01, max_iterations=500))

    # Train all models
    framework.train_all_models()

    # Verify all models are trained
    assert len(framework.trained_models) == 3

    # Evaluate all models
    results = framework.evaluate_all_models()

    # Verify results for all models
    assert len(results) == 3
    for model_name in ['Linear Reg', 'Ridge Reg', 'Lasso Reg']:
        assert model_name in results
        assert 'test_r2' in results[model_name]
        assert 'test_rmse' in results[model_name]


def test_model_comparison_error_handling():
    """Test error handling in model comparison framework."""
    framework = ModelComparisonFramework()

    # Try to evaluate before training - should raise error
    with pytest.raises(RuntimeError, match="Models must be trained and evaluated before comparison"):
        framework.compare_models()

    # Try to get best model before evaluation - should raise error
    with pytest.raises(RuntimeError, match="Models must be trained and evaluated first"):
        framework.get_best_model('test_r2')