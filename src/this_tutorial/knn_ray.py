# knn_ray + object store + batches
import logging
from pathlib import Path

import polars as pl
import numpy as np
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


N_POINTS = 10
LIMIT = 10
DEFAULT_K = 4


def calculate_distances(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """
    Calculate mutual Euclidean distances between M query and N reference points.

    Parameters:
    ----------
    query_points: np.ndarray
        (M, 3) array of query points
    reference_points: np.ndarray
        (N, 3+) array of reference points

    Returns:
    --------
    distances: np.ndarray
        (M, N) array of the distances
    """
    # Expand for broadcasting
    query_points = query_points[:, np.newaxis,:3]
    reference_points = reference_points[np.newaxis, :, :3]
    return np.sqrt(np.sum((reference_points - query_points) ** 2, axis=-1))


def knn_search(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    k: int,
):
    """
    Find k nearest neighbour reference point indices for N query points.

    Returns:
    --------
    indices: np.ndarray
        (N, k) matrix of integral indices
    """
    distances = calculate_distances(query_points, reference_points).T
    return np.argpartition(distances, k, axis=0)[:k].T


def create_grid(n_points: int = N_POINTS) -> tuple[np.ndarray, ...]:
    """
    Create a homogenous grid of points to create a map.

    Returns:
    --------
    x: np.ndarray
        Flattened (n_points x n_points,) array of x values
    y: np.ndarray
        Flattened (n_points x n_points,) array of x values
    """
    # Note: Tested indirectly via `create_query_points`
    # TODO: Add floor
    x = np.linspace(-LIMIT, LIMIT, n_points)
    y = np.linspace(-LIMIT, LIMIT, n_points)
    return tuple(arr.flatten() for arr in np.meshgrid(x, y))


def create_query_points(n_points: int = N_POINTS, floor: int = 1) -> np.ndarray:
    """
    Create a homogenous grid of points with a floor to create a map.

    Returns:
    --------
    query_points: np.ndarray
        (n_points x n_points, 3) array of query points
    """
    x, y = create_grid(n_points=n_points)
    return np.vstack([x, y, np.ones(x.shape[0]) * floor]).T


def compute_prices(query_points, reference_points, k: int = DEFAULT_K):
    """
    Find prices for N data_points.

    Parameters:
    ----------
    query_points: np.ndarray
        (N, 3) array of query points
    reference_points: np.ndarray
        (M, 4) array of data points with x, y, floor, and price
    k: int
        Number of nearest neighbors to consider

    Returns:
    --------
    prices: np.ndarray
        (N,) array of prices
    """
    indices = knn_search(query_points, reference_points, k)
    prices: np.ndarray = reference_points[indices, 3]
    return prices.mean(axis=1)


def load_data_points(path: Path = Path("data.parquet")) -> np.ndarray:
    """
    Load reference data points from a Parquet file.

    Returns:
    --------
    data_points: np.ndarray
        (N, 4) array of data points with x, y, floor, and price columns
    """

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
    ray.init()

    data_points = load_data_points()
    data_points_ref = ray.put(data_points)  # Store this

    query_points = create_query_points(1000)
    batch_size = 1000

    query_point_batches = [
        query_points[:, i : i + batch_size]
        for i in range(0, query_points.shape[1] // batch_size * batch_size, batch_size)
    ]
    logger.info(f"Submitting {len(query_point_batches)} batches of query points")

    # Compute prices for the query points
    prices = ray.get(
        [
            compute_prices.remote(query_point_batch, data_points_ref)
            for query_point_batch in query_point_batches
        ]
    )
    prices = np.concatenate(prices)
    output_df = pl.DataFrame(
        {
            "x": query_points[0],
            "y": query_points[1],
            "floor": query_points[2],
            "price": prices,
        }
    )
    print(output_df)
