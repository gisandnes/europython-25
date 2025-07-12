import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from this_tutorial import knn_jax


@pytest.fixture
def jax_device():
    return jax.devices("cpu")[0]


@pytest.fixture
def regression_data(jax_device: jax.Device) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate sample regression data for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    # Create a simple target function: sum of features + noise
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
    return jnp.array(X, device=jax_device), jnp.array(y, device=jax_device)


def test_knn_jax_matches_sklearn(dataset, query_points, k, jax_device):
    """Test that our custom JAX KNN implementation matches sklearn's results."""
    # Convert numpy arrays to JAX arrays
    dataset_jax = jnp.array(dataset, device=jax_device)
    query_points_jax = jnp.array(query_points, device=jax_device)

    # Fit the sklearn model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(dataset)

    # Find neighbors of query points using sklearn
    distances, indices = nbrs.kneighbors(query_points)

    # Find neighbors using our custom JAX implementation
    test_indices_jax = knn_jax.knn_search(query_points_jax, dataset_jax, k)
    # Convert JAX array back to numpy for comparison
    test_indices_np = np.array(test_indices_jax)

    # Compare results - sort both to handle different ordering
    sklearn_sorted = np.sort(indices, axis=1)
    jax_sorted = np.sort(test_indices_np, axis=1)

    # Assert that the results match
    assert np.array_equal(sklearn_sorted, jax_sorted), (
        "Custom JAX KNN implementation doesn't match sklearn results"
    )

    # Calculate only the distances we need using our custom JAX implementation
    # For each query point, calculate distances to all its k nearest neighbors at once
    test_distances_np = np.zeros_like(distances)
    for i in range(query_points.shape[0]):
        # Get all k neighbor points for this query point
        neighbor_points = dataset[test_indices_np[i, :]]  # Shape: (k, dims)
        # Calculate distances to all neighbors at once using broadcasting
        test_distances_np[i, :] = np.sqrt(np.sum((query_points[i] - neighbor_points) ** 2, axis=1))

    # Sort distances for comparison (since neighbor order might differ)
    sklearn_distances_sorted = np.sort(distances, axis=1)
    jax_distances_sorted = np.sort(test_distances_np, axis=1)

    # Assert that the distances match (with some tolerance for floating point precision)
    assert np.allclose(sklearn_distances_sorted, jax_distances_sorted, rtol=1e-5), (
        f"Distance calculations don't match: sklearn={sklearn_distances_sorted}, jax={jax_distances_sorted}"
    )


class TestKNNRegressor:
    """Tests for the KNNRegressor class."""

    def test_initialization(self):
        """Test that KNNRegressor initializes correctly."""
        regressor = knn_jax.KNNRegressor(k=3)
        assert regressor.k == 3
        assert regressor.distances_func == knn_jax.euclidean_distances
        assert regressor.X_train_ is None
        assert regressor.y_train_ is None

    def test_predict_without_fit_raises_error(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test that predict raises error when called before fit."""
        X, _ = regression_data
        regressor = knn_jax.KNNRegressor()

        with pytest.raises(ValueError, match="This KNNRegressor instance is not fitted yet"):
            regressor.predict(X[:5])

    def test_basic_functionality(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test basic fit and predict functionality."""
        X, y = regression_data

        # Split data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        # Fit and predict
        regressor = knn_jax.KNNRegressor(k=5)
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        # Check output shape
        assert predictions.shape == (len(X_test),)

        # Check that predictions are reasonable (should be within a reasonable range)
        assert jnp.all(jnp.isfinite(predictions))
        assert predictions.min() >= y_train.min() - 1  # Some tolerance
        assert predictions.max() <= y_train.max() + 1  # Some tolerance

    def test_single_prediction(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test prediction with a single query point."""
        X, y = regression_data

        regressor = knn_jax.KNNRegressor(k=3)
        regressor.fit(X, y)

        # Predict single point
        single_pred = regressor.predict(X[:1])
        assert single_pred.shape == (1,)

    def test_api_compatibility_with_sklearn(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test that our KNNRegressor has sklearn-compatible API."""
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        # Test our implementation
        our_regressor = knn_jax.KNNRegressor(k=5)
        our_regressor.fit(X_train, y_train)
        our_predictions = our_regressor.predict(X_test)

        # Test sklearn implementation
        sklearn_regressor = KNeighborsRegressor(n_neighbors=5, algorithm="brute")
        sklearn_regressor.fit(X_train, y_train)
        sklearn_predictions = sklearn_regressor.predict(X_test)

        # Check that both work and produce reasonable results
        assert our_predictions.shape == sklearn_predictions.shape
        assert jnp.all(jnp.isfinite(our_predictions))
        assert np.all(np.isfinite(sklearn_predictions))

    def test_numerical_similarity_to_sklearn(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test that our implementation gives numerically similar results to sklearn."""
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        # Our implementation
        our_regressor = knn_jax.KNNRegressor(k=5)
        our_regressor.fit(X_train, y_train)
        our_predictions = np.array(our_regressor.predict(X_test))

        # Sklearn implementation
        sklearn_regressor = KNeighborsRegressor(n_neighbors=5, algorithm="brute")
        sklearn_regressor.fit(np.asarray(X_train), np.asarray(y_train))
        sklearn_predictions = sklearn_regressor.predict(np.asarray(X_test))

        # Check numerical similarity
        np.testing.assert_allclose(
            our_predictions,
            sklearn_predictions,
            rtol=1e-6,  # Reasonable tolerance for float32 vs float64 precision differences
            atol=1e-6,
            err_msg="Our KNN regressor predictions don't match sklearn",
        )

    def test_different_k_values(self, regression_data: tuple[jnp.ndarray, jnp.ndarray]):
        """Test that different k values work correctly."""
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        for k in [1, 3, 5, 10]:
            regressor = knn_jax.KNNRegressor(k=k)
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)

            assert predictions.shape == (len(X_test),)
            assert jnp.all(jnp.isfinite(predictions))

    def test_k_equals_one_exact_match(self, jax_device: jax.Device):
        """Test that k=1 gives exact match when query point is in training set."""
        # Simple test data where we know the answer
        X_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=jax_device)
        y_train = jnp.array([10.0, 20.0, 30.0], device=jax_device)

        regressor = knn_jax.KNNRegressor(k=1)
        regressor.fit(X_train, y_train)

        # Query with training points - should get exact matches
        predictions = regressor.predict(X_train)
        np.testing.assert_allclose(np.array(predictions), np.array(y_train), rtol=1e-10)

    @pytest.mark.xfail(reason="Not working on all devices")
    def test_handles_jax_and_numpy_arrays(self, regression_data):
        """Test that the regressor handles both JAX and NumPy arrays."""
        X, y = regression_data
        X_train, X_test = np.asarray(X[:80]), np.asarray(X[80:])
        y_train = np.asarray(y[:80])

        regressor = knn_jax.KNNRegressor(k=5)

        # Fit with NumPy arrays
        regressor.fit(X_train, y_train)

        # Predict with JAX arrays
        X_test_jax = jnp.array(X_test)
        predictions_jax = regressor.predict(X_test_jax)

        # Predict with NumPy arrays
        predictions_np = regressor.predict(X_test)

        # Results should be the same
        np.testing.assert_allclose(np.array(predictions_jax), np.array(predictions_np), rtol=1e-12)
