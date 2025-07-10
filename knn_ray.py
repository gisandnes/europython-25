import numpy as np
import ray
import polars as pl
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_distances(query_points: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Calculate mutual distances between M query and N reference points.

    Returns:
    --------
    distances: np.ndarray
        (N, M) array of the distances
    """
    # Expand for broadcasting
    query_points = query_points[:, :, np.newaxis]
    dataset = dataset[:3, np.newaxis]
    return np.sqrt(np.sum((dataset - query_points) ** 2, axis=0))


N_POINTS = 10
LIMIT = 10
DEFAULT_K = 4


# @ray.remote
def knn_search(
    query_points: np.ndarray,
    dataset: np.ndarray,
    k: int,
    distances_func=calculate_distances,
):
    """
    Find k nearest neighbour reference point indices for N query points.

    Returns:
    --------
    indices: np.ndarray
        (N, k) matrix of integral indices

    """
    distances = distances_func(query_points, dataset).T
    nearest_indices = np.argpartition(distances, k, axis=0)[:k].T
    return nearest_indices


def create_grid() -> tuple[np.ndarray, ...]:
    """
    Create a homogenous grid of points to create a map.

    Returns:
    --------
    x: np.ndarray
        Flattened (N_POINTS x N_POINTS,) array of x values
    y: np.ndarray
        Flattened (N_POINTS x N_POINTS,) array of x values
    """
    # TODO: Add floor
    x = np.linspace(-LIMIT, LIMIT, N_POINTS)
    y = np.linspace(-LIMIT, LIMIT, N_POINTS)
    return tuple(arr.flatten() for arr in np.meshgrid(x, y))


@ray.remote
def compute_prices(query_points, data_points):
    """
    Find prices for N data_points.

    Returns:
    --------
    prices: np.ndarray
        (N,) array of prices
    """
    indices = knn_search(query_points, data_points, DEFAULT_K)
    prices: np.ndarray = data_points[3][indices]
    return prices.mean(axis=1)


def load_data_points(path: Path = Path("data.parquet")) -> np.ndarray:
    df = pl.read_parquet(path)
    return np.vstack(
        [
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["floor"].to_numpy(),
            df["price"].to_numpy(),
        ]
    )


if __name__ == "__main__":
    data_points = load_data_points()

    x, y = create_grid()
    query_points = np.vstack([x, y, np.ones(x.shape[0])])

    ray.init()

    # TODO: Do this in batches
    prices_task = compute_prices.remote(query_points, data_points)
    prices = ray.get(prices_task)
    output_df = pl.DataFrame(
        {
            "x": query_points[0],
            "y": query_points[1],
            "floor": query_points[2],
            "price": prices,
        }
    )
    output_df.write_parquet("grid.parquet")
    logger.info("Saved grid.parquet")
