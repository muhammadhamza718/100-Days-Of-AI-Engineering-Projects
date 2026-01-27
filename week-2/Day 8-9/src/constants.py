"""
Constants and type definitions for the regression system.

This module defines global constants and type aliases used throughout
the regression implementation to ensure consistency and maintainability.
"""

from typing import Union, Tuple, List
import numpy as np

# Mathematical Constants
EPSILON = 1e-15  # Small value to prevent division by zero
SQRT_EPSILON = 1e-8  # Small value for square root operations
MIN_LEARNING_RATE = 1e-8  # Minimum allowed learning rate
MAX_LEARNING_RATE = 1.0   # Maximum allowed learning rate
DEFAULT_DEGREE = 2        # Default polynomial degree
MAX_DEGREE = 10           # Maximum allowed polynomial degree
MIN_ITERATIONS = 10       # Minimum number of iterations
MAX_ITERATIONS = 100000   # Maximum number of iterations

# Model Types
MODEL_LINEAR = 'linear'
MODEL_POLYNOMIAL = 'polynomial'
MODEL_RIDGE = 'ridge'
MODEL_LASSO = 'lasso'

# Regularization Types
REGULARIZATION_L1 = 'l1'  # Lasso
REGULARIZATION_L2 = 'l2'  # Ridge
REGULARIZATION_ELASTIC_NET = 'elastic_net'

# Gradient Descent Types
GD_BATCH = 'batch'
GD_STOCHASTIC = 'stochastic'
GD_MINI_BATCH = 'mini_batch'

# Feature Scaling Methods
SCALING_STANDARDIZE = 'standardize'
SCALING_NORMALIZE = 'normalize'
SCALING_MIN_MAX = 'min_max'

# Data Types
Vector = Union[List[float], np.ndarray]
Matrix = Union[List[List[float]], np.ndarray]
ArrayLike = Union[List, np.ndarray]

# Common Hyperparameter Ranges
LEARNING_RATE_RANGE = (1e-6, 1.0)
LAMBDA_REG_RANGE = (0.0, 10.0)
DEGREE_RANGE = (1, 10)
ITERATION_RANGE = (10, 100000)

# Error Messages
MSG_INVALID_SHAPE = "Invalid array shape"
MSG_FEATURE_MISMATCH = "Feature dimension mismatch"
MSG_NOT_FITTED = "Model must be fitted before making predictions"
MSG_CONVERGENCE_FAILED = "Model failed to converge within maximum iterations"
MSG_INVALID_INPUT = "Invalid input data"
MSG_NEGATIVE_VALUE = "Expected non-negative value"
MSG_NONEMPTY_REQUIRED = "Expected non-empty array"

# Warning Messages
WARN_FEATURE_NOT_SCALED = "Features are not scaled, which may affect convergence"
WARN_HIGH_LEARNING_RATE = "High learning rate may cause divergence"
WARN_LOW_LEARNING_RATE = "Low learning rate may cause slow convergence"
WARN_INSUFFICIENT_DATA = "Insufficient data for reliable model training"

# Default Values
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_TOLERANCE = 1e-6
DEFAULT_LAMBDA_REG = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_STATE = 42