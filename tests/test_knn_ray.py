import numpy as np
from numpy.testing import assert_allclose

from this_tutorial.knn_ray import create_grid, create_query_points, calculate_distances, knn_search


def test_create_grid():
    x, y = create_grid(n_points=5)
    assert x.shape == (25,)
    assert y.shape == (25,)
    assert x[0] == -10.0
    assert x[-1] == 10.0
    assert y[0] == -10.0
    assert y[-1] == 10.0


def test_create_query_points():
    query_points = create_query_points(n_points=5, floor=2)
    assert query_points.shape == (25, 3)
    assert np.all(query_points[:,2] == 2)  # Floor should be constant at 2
    assert np.all(query_points[:,0] >= -10) and np.all(query_points[:,0] <= 10)
    assert np.all(query_points[:,1] >= -10) and np.all(query_points[:,1] <= 10)
    assert_allclose(query_points[17], np.asarray([0., 5., 2.]))


def test_calculate_distances():
    query_points = np.array([[0, 0, 1], [1, 1, 1]])
    reference_points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [1, 1, 1]])
    
    distances = calculate_distances(query_points, reference_points)
    assert distances.shape == (2, 4)

    
    expected_distances = np.array([
        [1.0, np.sqrt(3), 3.0, np.sqrt(2)],
        [np.sqrt(3), 1.0, np.sqrt(3), 0.0]
    ])
    
    assert_allclose(distances, expected_distances)


def test_knn_search():
    query_points = np.array([[0, 0, 1], [3, 3, 3]])
    reference_points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [1, 1, 1]])
    
    k = 2
    indices = knn_search(query_points, reference_points, k)
    
    assert indices.shape == (2, k)
    expected_indices = np.array([[0, 3], [2, 3]])

    assert np.array_equal(indices, expected_indices) 