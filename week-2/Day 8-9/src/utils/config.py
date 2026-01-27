"""
Configuration utilities for the regression system.

This module provides centralized configuration and logging utilities
for the regression implementation.
"""

import logging
from typing import Dict, Any


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Set up logging for the regression system.

    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string level to logging constant
    level_constant = getattr(logging, level.upper())

    # Create and configure logger
    logger = logging.getLogger('regression_system')
    logger.setLevel(level_constant)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_constant)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def get_default_hyperparameters() -> Dict[str, Any]:
    """
    Get default hyperparameters for regression models.

    Returns:
        Dict[str, Any]: Dictionary containing default hyperparameters
    """
    return {
        # Learning rate for gradient descent
        'learning_rate': 0.01,

        # Maximum iterations for training
        'max_iterations': 1000,

        # Convergence tolerance
        'tolerance': 1e-6,

        # Regularization parameter (lambda)
        'lambda_reg': 0.01,

        # Polynomial degree for polynomial regression
        'degree': 2,

        # Batch size for mini-batch gradient descent
        'batch_size': 32,

        # Whether to normalize features
        'normalize_features': True
    }


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize hyperparameters.

    Args:
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters to validate

    Returns:
        Dict[str, Any]: Validated and normalized hyperparameters

    Raises:
        ValueError: If hyperparameters are invalid
    """
    validated_params = {}

    # Validate learning rate
    lr = hyperparams.get('learning_rate', 0.01)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"Learning rate must be a positive number, got {lr}")
    validated_params['learning_rate'] = float(lr)

    # Validate max iterations
    max_iter = hyperparams.get('max_iterations', 1000)
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(f"Max iterations must be a positive integer, got {max_iter}")
    validated_params['max_iterations'] = int(max_iter)

    # Validate tolerance
    tol = hyperparams.get('tolerance', 1e-6)
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError(f"Tolerance must be a positive number, got {tol}")
    validated_params['tolerance'] = float(tol)

    # Validate regularization parameter
    lambda_reg = hyperparams.get('lambda_reg', 0.01)
    if not isinstance(lambda_reg, (int, float)) or lambda_reg < 0:
        raise ValueError(f"Regularization parameter must be non-negative, got {lambda_reg}")
    validated_params['lambda_reg'] = float(lambda_reg)

    # Validate polynomial degree
    degree = hyperparams.get('degree', 2)
    if not isinstance(degree, int) or degree < 1:
        raise ValueError(f"Polynomial degree must be a positive integer, got {degree}")
    validated_params['degree'] = int(degree)

    # Validate batch size
    batch_size = hyperparams.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}")
    validated_params['batch_size'] = int(batch_size)

    # Validate normalize_features
    norm_feat = hyperparams.get('normalize_features', True)
    if not isinstance(norm_feat, bool):
        raise ValueError(f"Normalize features must be boolean, got {norm_feat}")
    validated_params['normalize_features'] = bool(norm_feat)

    return validated_params


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get recommended configuration for a specific model type.

    Args:
        model_type (str): Type of model ('linear', 'polynomial', 'ridge', 'lasso')

    Returns:
        Dict[str, Any]: Recommended configuration for the model type
    """
    configs = {
        'linear': {
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6
        },
        'polynomial': {
            'learning_rate': 0.001,  # Lower LR for polynomial features
            'max_iterations': 2000,  # More iterations for complex features
            'tolerance': 1e-6,
            'degree': 2
        },
        'ridge': {
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'lambda_reg': 0.1
        },
        'lasso': {
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'lambda_reg': 0.1
        }
    }

    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: {list(configs.keys())}")

    return configs[model_type]