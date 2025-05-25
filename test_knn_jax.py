import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

import knn_jax


@pytest.fixture
def jax_device():
    return jax.devices("cpu")[0]


def test_knn_jax_matches_sklearn(dataset, query_points, k, jax_device):
    """Test that our custom JAX KNN implementation matches sklearn's results."""
    # Convert numpy arrays to JAX arrays
    dataset_jax = jnp.array(dataset, device=jax_device)
    query_points_jax = jnp.array(query_points, device=jax_device)

    # Fit the sklearn model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(dataset)

    # Find neighbors of query points using sklearn
    distances, indices = nbrs.kneighbors(query_points)

    # Find neighbors using our custom JAX implementation
    test_indices_jax = knn_jax.knn_search(query_points_jax, dataset_jax, k)
    # Convert JAX array back to numpy for comparison
    test_indices_np = np.array(test_indices_jax)

    # Compare results - sort both to handle different ordering
    sklearn_sorted = np.sort(indices, axis=1)
    jax_sorted = np.sort(test_indices_np, axis=1)

    # Assert that the results match
    assert np.array_equal(sklearn_sorted, jax_sorted), (
        "Custom JAX KNN implementation doesn't match sklearn results"
    )

    # Calculate only the distances we need using our custom JAX implementation
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
    jax_distances_sorted = np.sort(test_distances_np, axis=1)

    # Assert that the distances match (with some tolerance for floating point precision)
    assert np.allclose(sklearn_distances_sorted, jax_distances_sorted, rtol=1e-5), (
        f"Distance calculations don't match: sklearn={sklearn_distances_sorted}, jax={jax_distances_sorted}"
    )
