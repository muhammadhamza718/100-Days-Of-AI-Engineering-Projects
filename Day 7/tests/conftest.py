"""
Test configuration and fixtures for the mathematics for ML project
"""
import pytest
import numpy as np


@pytest.fixture
def sample_vector():
    """Sample vector for testing"""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def sample_matrix():
    """Sample matrix for testing"""
    return np.array([[1.0, 2.0], [3.0, 4.0]])