# Baseline kNN Implementation in NumPy

import numpy as np


def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((dataset[:, np.newaxis, :] - query_points) ** 2, axis=-1))


def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
) -> np.ndarray:
    """
    Finds the k nearest neighbors for a single query point using NumPy.

    Args:
        query_point (np.ndarray): A single point (1D array).
        dataset (np.ndarray): The dataset of points (2D array).
        k (int): The number of neighbors to find.

    Returns:
        np.ndarray: Indices of the k nearest neighbors in the dataset.
    """
    # Calculate Euclidean distances from the query point to all dataset points
    # Broadcasting (dataset - query_point) subtracts query_point from each row of dataset
    # distances = np.sqrt(np.sum((dataset - query_points)**2, axis=-1))

    # This should work for multiple query points - could be an exercise for the workshop
    distances = distances_func(query_points, dataset)

    # Find the indices of the k smallest distances
    # np.argsort returns the indices that would sort the array
    # nearest_indices = np.argsort(distances, 0).T[:, :k]
    nearest_indices = np.argpartition(distances, k, axis=0)[:k].T
    return nearest_indices


def knn_batch(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    batch_count: int = 10,
    knn_search_func=knn_search,
    distances_func=calculate_distances,
) -> np.ndarray:
    """Processes multiple query points sequentially using NumPy."""
    results = []
    for query_point_batch in np.array_split(query_points, batch_count):
        indices = knn_search_func(
            query_point_batch, dataset, k, distances_func=distances_func
        )
        results.extend(indices)
    return np.stack(results)
