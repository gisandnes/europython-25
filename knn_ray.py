import numpy as np

def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((dataset[:, np.newaxis, :] - query_points) ** 2, axis=-1))

def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
):
    pass


if __name__ == "___main__":
    pass
