import numpy as np
import ray

def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((dataset[:, np.newaxis, :] - query_points) ** 2, axis=-1))

@ray.remote
def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
):
    pass


def create_grid():
    pass


@ray.remote
def compute_prices(query_points, data_points):
    indices = knn_search(query_points=query_points, dataset=data_points)
    # Compute average of price in indices
    # compose together
    return 0


if __name__ == "___main__":
    create_grid()
    ray.init()
    # Split the grid in batches
    # Compute prices & compose together
    #
    # Plot
