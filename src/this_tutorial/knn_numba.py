import numba
import numpy as np


@numba.njit
def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    M = query_points.shape[0]
    N = dataset.shape[0]
    D = query_points.shape[1]
    distances = np.empty((N, M), dtype=dataset.dtype)
    for i in range(N):
        for j in range(M):
            d = 0.0
            for k in range(D):
                tmp = dataset[i, k] - query_points[j, k]
                d += tmp * tmp
            distances[i, j] = np.sqrt(d)
    return distances


@numba.njit
def partition(arr, idx, left, right, pivot_index):
    pivot_value = arr[idx[pivot_index]]
    # Move pivot to end
    idx[pivot_index], idx[right] = idx[right], idx[pivot_index]
    store_index = left
    for i in range(left, right):
        if arr[idx[i]] < pivot_value:
            idx[store_index], idx[i] = idx[i], idx[store_index]
            store_index += 1
    # Move pivot to its final place
    idx[right], idx[store_index] = idx[store_index], idx[right]
    return store_index


@numba.njit
def quickselect_k_indices(arr, k):
    n = arr.shape[0]
    idx = np.arange(n)
    left = 0
    right = n - 1
    while True:
        if left == right:
            break
        pivot_index = left + (right - left) // 2
        pivot_new_index = partition(arr, idx, left, right, pivot_index)
        if k == pivot_new_index:
            break
        elif k < pivot_new_index:
            right = pivot_new_index - 1
        else:
            left = pivot_new_index + 1
    return idx[:k]


@numba.njit
def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Finds the k nearest neighbors for query points using Numba.

    Args:
        query_points (np.ndarray): Query points (2D array).
        dataset (np.ndarray): The dataset of points (2D array).
        k (int): The number of neighbors to find.

    Returns:
        np.ndarray: Indices of the k nearest neighbors in the dataset for each query point.
    """
    # Calculate distances from all query points to all dataset points
    distances = calculate_distances(query_points, dataset)

    # Find k nearest neighbors for each query point
    M = query_points.shape[0]
    nearest_indices = np.empty((M, k), dtype=np.int64)

    for i in range(M):
        # Get distances for this query point
        query_distances = distances[:, i]
        # Find indices of k smallest distances
        nearest_indices[i] = quickselect_k_indices(query_distances, k)

    return nearest_indices
