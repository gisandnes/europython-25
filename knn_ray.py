import numpy as np
import ray
import polars as pl
from pathlib import Path


def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    # Expand for broadcasting
    query_points = query_points[:, np.newaxis]
    dataset = dataset[:3, :, np.newaxis]

    return np.sqrt(
        np.sum((dataset - query_points) ** 2, axis=0)
    )


N_POINTS = 10
LIMIT = 10


# @ray.remote
def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
):
    distances = distances_func(query_points, dataset)
    breakpoint()
    nearest_indices = np.argpartition(distances, k, axis=0)[:k].T
    return nearest_indices


def create_grid() -> tuple[np.ndarray, ...]:
    """
    Create a homogenous grid of points to create a map.

    Returns:
    --------

    """
    # TODO: Add floor
    x = np.linspace(-LIMIT, LIMIT, N_POINTS)
    y = np.linspace(-LIMIT, LIMIT, N_POINTS)
    return tuple(
        arr.flatten() for arr in
        np.meshgrid(x, y)
    )


# @ray.remote
def compute_prices(query_points, data_points):
    indices = knn_search(query_points, data_points, 4)
    # indices = knn_search.remote(query_points, data_points, 4)
    # Compute average of price in indices
    # compose together
    return 0


def load_data_points(path: Path = Path("data.parquet")) -> np.ndarray:
    df = pl.read_parquet(path)
    return np.vstack([
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        df["floor"].to_numpy(),
        df["price"].to_numpy()
    ])


if __name__ == "__main__":
    # ray.init()
    data_points = load_data_points()

    x, y = create_grid()
    query_points = np.vstack([x, y, np.ones(x.shape[0])])

    #ray.init()
    f = compute_prices(query_points, data_points)
    # f = compute_prices.remote(query_points, data_points)
    # ray.get(f)


    # Split the grid in batches
    # Compute prices & compose together
    #
    # Plot
