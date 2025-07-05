import numpy as np
import ray

def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((dataset[:, np.newaxis, :] - query_points) ** 2, axis=-1))


N_POINTS = 10
LIMIT = 10


@ray.remote
def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
):
    distances = distances_func(query_points, dataset)
    nearest_indices = np.argpartition(distances, k, axis=0)[:k].T
    return nearest_indices


def create_grid() -> tuple[np.ndarray, ...]:
    """
    Create a homogenous grid of points to create a map.
    """
    # TODO: Add floor
    x = np.linspace(-LIMIT, LIMIT, N_POINTS)
    y = np.linspace(-LIMIT, LIMIT, N_POINTS)
    return np.meshgrid(x, y)


@ray.remote
def compute_prices(query_points, data_points):
    indices = knn_search(query_points=query_points, dataset=data_points)
    # Compute average of price in indices
    # compose together
    return 0


if __name__ == "___main__":
    ray.init()

    x, y = create_grid()
    data = np.hstack([x, y, 1])

    # Split the grid in batches
    # Compute prices & compose together
    #
    # Plot
