from collections.abc import Callable

import jax.lax
import jax.numpy as jnp
import numpy as np


def euclidean_distances_no_jit(
    query_points: jnp.ndarray, dataset: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the Euclidean distance between a set of query points and a dataset of points.

    Args:
        query_points (jnp.ndarray): Array of shape (n_queries, n_features).
        dataset (jnp.ndarray): Array of shape (n_samples, n_features).

    Returns:
        jnp.ndarray: The Euclidean distance between the query points and the dataset.
    """
    # Broadcasting (dataset - query_point) subtracts query_point from each row of dataset
    return jnp.sqrt(jnp.sum((dataset[:, jnp.newaxis, :] - query_points) ** 2, axis=-1))


euclidean_distances = jax.jit(euclidean_distances_no_jit)


def knn_search_no_jit(
    query_points: jnp.ndarray,
    dataset: jnp.ndarray,
    k: int,
    distances_func: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = euclidean_distances,
) -> jnp.ndarray:
    """
    Finds the k nearest neighbors for a single query point using NumPy.

    Args:
        query_points (jnp.ndarray): Array of shape (n_queries, n_features).
        dataset (jnp.ndarray): Array of shape (n_samples, n_features).
        k (int): The number of neighbors to find.

    Returns:
        jnp.ndarray: Array of shape (n_queries, k) containing the indices of the k nearest neighbors in the dataset.
    """
    # This should work for multiple query points - could be an exercise for the workshop
    distances = distances_func(query_points, dataset)

    # Find the indices of the k smallest distances
    values, nearest_indices = jax.lax.top_k(-distances.T, k)
    # nearest_indices = jnp.argpartition(distances, k, axis=0)[:k].T
    return nearest_indices


knn_search = jax.jit(knn_search_no_jit, static_argnames=["k", "distances_func"])


def knn_batch(
    query_points: jnp.ndarray,
    dataset: jnp.ndarray,
    k: int,
    batch_count: int = 10,
    knn_search_func=knn_search,
    distances_func=euclidean_distances,
) -> jnp.ndarray:
    """Processes multiple query points sequentially using NumPy."""
    results = []
    for query_point_batch in jnp.array_split(query_points, batch_count):
        indices = knn_search_func(
            query_point_batch, dataset, k, distances_func=distances_func
        )
        results.extend(indices)
    return jnp.stack(results)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor with scikit-learn compatible API using JAX.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use for prediction.
    distances_func : callable, default=calculate_distances
        Function to calculate distances between points.
    """

    def __init__(
        self,
        k: int = 5,
        distances_func: Callable[
            [jnp.ndarray, jnp.ndarray], jnp.ndarray
        ] = euclidean_distances,
    ) -> None:
        self.k = k
        self.distances_func = distances_func
        self.X_train_: jnp.ndarray | None = None
        self.y_train_: jnp.ndarray | None = None

    def fit(
        self, X: jnp.ndarray | np.ndarray, y: jnp.ndarray | np.ndarray
    ) -> "KNNRegressor":
        """
        Fit the KNN regressor by storing the training data.

        Parameters
        ----------
        X : jnp.ndarray of shape (n_samples, n_features)
            Training data features.
        y : jnp.ndarray of shape (n_samples,)
            Training data targets.

        Returns
        -------
        self : KNNRegressor
            Returns self for method chaining.
        """
        self.X_train_ = jnp.asarray(X)
        self.y_train_ = jnp.asarray(y)
        return self

    def predict(self, X: jnp.ndarray | np.ndarray) -> jnp.ndarray:
        """
        Predict target values for the provided data.

        Parameters
        ----------
        X : jnp.ndarray of shape (n_samples, n_features)
            Query points to predict.

        Returns
        -------
        jnp.ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError(
                "This KNNRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X = jnp.asarray(X)

        # Find k nearest neighbours for each query point
        neighbor_indices = knn_search(
            X, self.X_train_, self.k, distances_func=self.distances_func
        )

        # Get the target values of the nearest neighbours
        neighbor_targets = self.y_train_[neighbor_indices]

        # Return the mean of the neighbour targets (regression prediction)
        return jnp.mean(neighbor_targets, axis=1)
