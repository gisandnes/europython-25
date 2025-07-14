# knn_ray + object store + batches
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


N_POINTS = 10   # Default number of points in each dimension for the grid
LIMIT = 10.0    # +/- Span of the grid
DEFAULT_K = 4   # How many nearest neighbors to consider


def calculate_distances(query_points: np.ndarray, reference_points: np.ndarray, *, n_dim: int = 3) -> np.ndarray:
    """
    Calculate mutual Euclidean distances between M query and N reference points.

    Parameters:
    ----------
    query_points: np.ndarray
        (M, n_dim+) array of query points
    reference_points: np.ndarray
        (N, n_dim+) array of reference points
    n_dim: int
        Number of dimensions to consider (default: 3, for x, y, floor)

    Returns:
    --------
    distances: np.ndarray
        (M, N) array of the distances
    """
    # Expand for broadcasting
    query_points = query_points[:, np.newaxis,:n_dim]
    reference_points = reference_points[np.newaxis, :, :n_dim]
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


def load_reference_points(path: Path = Path("data.parquet")) -> np.ndarray:
    """
    Load reference data points from a Parquet file.

    Returns:
    --------
    data_points: np.ndarray
        (N, 4) array of data points with x, y, floor, and price columns
    """

    df = pd.read_parquet(path)
    return df[["x", "y", "floor", "price"]].to_numpy().astype(float)


def combine_points_and_prices(
    query_points: np.ndarray, prices: np.ndarray
) -> pd.DataFrame:
    """
    Prepare human-friendly output from numpy arrays.

    Parameters:
    ----------
    query_points: np.ndarray
        (N, 3) array of query points
    prices: np.ndarray
        (N,) array of prices

    Returns:
    --------
    df: pd.DataFrame
        DataFrame with columns x, y, floor, price
    """
    return pd.DataFrame(
        {
            "x": query_points[:,0],
            "y": query_points[:,1],
            "floor": query_points[:,2],
            "price": prices,
        }
    )


compute_prices_ray = ray.remote(compute_prices)


if __name__ == "__main__":
    ray.init()

    data_points = load_reference_points()
    data_points_ref = ray.put(data_points)  # Store this

    query_points = create_query_points(1000)
    batch_size = 1000

    query_point_batches = [
        query_points[i : i + batch_size, :]
        for i in range(0, query_points.shape[0] // batch_size * batch_size, batch_size)
    ]
    logger.info(f"Submitting {len(query_point_batches)} batches of query points")

    # Compute prices for the query points
    prices = ray.get(
        [
            compute_prices_ray.remote(query_point_batch, data_points_ref)
            for query_point_batch in query_point_batches
        ]
    )
    prices = np.concatenate(prices)
    output_df = combine_points_and_prices(
        query_points=query_points,
        prices=prices,
    )
    print(output_df)
