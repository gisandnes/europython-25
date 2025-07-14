import numpy as np
import pytest


@pytest.fixture
def dataset(n_dataset, dims):
    return np.random.rand(n_dataset, dims).astype(np.float32)


@pytest.fixture
def query_points(m_query, dims):
    return np.random.rand(m_query, dims).astype(np.float32)


@pytest.fixture
def n_dataset():
    """Number of points in the dataset"""
    return 100


@pytest.fixture
def m_query():
    """Number of query points"""
    return 10


@pytest.fixture
def dims():
    """Number of dimensions"""
    return 2


@pytest.fixture
def k():
    """Number of neighbors to find"""
    return 5
