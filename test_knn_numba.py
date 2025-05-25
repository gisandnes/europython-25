import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

import knn_numba
from knn_numba import partition, quickselect_k_indices


@pytest.mark.parametrize(
    "arr, left, right, pivot_index, expected_partition",
    [
        # Simple case
        (np.array([3, 1, 2]), 0, 2, 0, [1, 2, 3]),
        # Already sorted
        (np.array([1, 2, 3]), 0, 2, 1, [1, 2, 3]),
        # Reverse sorted
        (np.array([3, 2, 1]), 0, 2, 2, [1, 2, 3]),
        # With duplicates
        (np.array([2, 1, 2]), 0, 2, 0, [1, 2, 2]),
    ],
)
def test_partition(arr, left, right, pivot_index, expected_partition):
    arr = arr.copy()
    idx = np.arange(len(arr))
    # Partition modifies idx in-place
    partition(arr, idx, left, right, pivot_index)
    partitioned = arr[idx]
    assert set(partitioned) == set(expected_partition)
    # Check that all elements are present
    assert sorted(partitioned) == sorted(expected_partition)


@pytest.mark.parametrize(
    "arr, k",
    [
        (np.array([3, 1, 2]), 2),
        (np.array([10, 5, 8, 1, 7]), 3),
        (np.array([1, 2, 3, 4, 5]), 4),
        (np.array([5, 4, 3, 2, 1]), 1),
        (np.array([2, 2, 2, 2]), 2),
        (np.random.sample((1000,)), 10),
    ],
)
def test_quickselect_k_indices(arr, k):
    arr = arr.copy()
    idx = quickselect_k_indices(arr, k)
    # The returned indices should correspond to the k smallest elements
    selected = arr[idx]
    expected = np.partition(arr, k)[:k]
    # Compare as sets (order not guaranteed)
    assert set(selected) == set(expected)
    # All indices should be unique and within bounds
    assert len(set(idx)) == len(idx)
    assert np.all(idx >= 0) and np.all(idx < len(arr))


def test_knn_numba_matches_sklearn(dataset, query_points, k):
    """Test that our custom Numba KNN implementation matches sklearn's results."""
    # Fit the sklearn model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(dataset)

    # Find neighbors of query points using sklearn
    distances, indices = nbrs.kneighbors(query_points)

    # Find neighbors using our custom Numba implementation
    test_indices_np = knn_numba.knn_search(query_points, dataset, k)

    # Compare results - sort both to handle different ordering
    sklearn_sorted = np.sort(indices, axis=1)
    numba_sorted = np.sort(test_indices_np, axis=1)

    # Assert that the results match
    assert np.array_equal(sklearn_sorted, numba_sorted), (
        "Custom Numba KNN implementation doesn't match sklearn results"
    )

    # Calculate only the distances we need using our custom Numba implementation
    # For each query point, calculate distances to all its k nearest neighbors at once
    test_distances_np = np.zeros_like(distances)
    for i in range(query_points.shape[0]):
        # Get all k neighbor points for this query point
        neighbor_points = dataset[test_indices_np[i, :]]  # Shape: (k, dims)
        # Calculate distances to all neighbors at once using broadcasting
        test_distances_np[i, :] = np.sqrt(
            np.sum((query_points[i] - neighbor_points) ** 2, axis=1)
        )

    # Sort distances for comparison (since neighbor order might differ)
    sklearn_distances_sorted = np.sort(distances, axis=1)
    numba_distances_sorted = np.sort(test_distances_np, axis=1)

    # Assert that the distances match (with some tolerance for floating point precision)
    assert np.allclose(sklearn_distances_sorted, numba_distances_sorted, rtol=1e-5), (
        f"Distance calculations don't match: sklearn={sklearn_distances_sorted}, numba={numba_distances_sorted}"
    )
