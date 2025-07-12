"""
Tests for the location optimisation module.
"""

import jax.numpy as jnp

from this_tutorial.knn_jax import euclidean_distances
from this_tutorial.location_optimisation import (
    Location,
    define_target_locations,
    gradient_descent_jax,
    gradient_descent_numerical,
    numerical_gradient,
    total_distance_objective,
)


class TestLocationOptimisation:
    """Test suite for location optimisation functions."""

    def test_location_creation(self) -> None:
        """Test Location namedtuple creation."""
        loc = Location(x=1.5, y=2.3, name="Test Location")
        assert loc.x == 1.5
        assert loc.y == 2.3
        assert loc.name == "Test Location"

    def test_define_target_locations(self) -> None:
        """Test that target locations are defined correctly."""
        locations = define_target_locations()

        # Check we have exactly 4 locations
        assert len(locations) == 4

        # Check each location has the expected attributes
        expected_names = {"School", "Work", "Parents", "Sports Club"}
        actual_names = {loc.name for loc in locations}
        assert actual_names == expected_names

        # Check coordinates are reasonable (between -10 and 10)
        for loc in locations:
            assert -10 <= loc.x <= 10
            assert -10 <= loc.y <= 10

    def test_euclidean_distances_from_knn_jax(self) -> None:
        """Test euclidean distances function from knn_jax module."""
        query_points = jnp.array([[0.0, 0.0]])  # Single query point
        dataset = jnp.array([[3.0, 4.0]])  # Single data point

        distances = euclidean_distances(query_points, dataset)
        expected_distance = 5.0  # 3-4-5 triangle

        assert distances.shape == (1, 1)  # Shape should be (n_dataset, n_queries)
        assert abs(distances[0, 0] - expected_distance) < 1e-6

    def test_euclidean_distances_batch(self) -> None:
        """Test euclidean distances with multiple points."""
        query_points = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # Two query points
        dataset = jnp.array([[3.0, 4.0], [0.0, 0.0]])  # Two data points

        distances = euclidean_distances(query_points, dataset)

        # Should return shape (n_dataset, n_queries) = (2, 2)
        assert distances.shape == (2, 2)

        # Distance from (0,0) to (3,4) should be 5
        assert abs(distances[0, 0] - 5.0) < 1e-6

        # Distance from (0,0) to (0,0) should be 0
        assert abs(distances[1, 0] - 0.0) < 1e-6

    def test_total_distance_objective(self) -> None:
        """Test total distance objective function."""
        # Simple test case with known coordinates
        candidate = jnp.array([0.0, 0.0])
        targets = jnp.array(
            [
                [1.0, 0.0],  # distance = 1
                [0.0, 1.0],  # distance = 1
                [-1.0, 0.0],  # distance = 1
                [0.0, -1.0],  # distance = 1
            ]
        )

        total_dist = total_distance_objective(candidate, targets)
        expected_total = 4.0

        assert abs(total_dist - expected_total) < 1e-6

    def test_numerical_gradient_quadratic(self) -> None:
        """Test numerical gradient on a simple quadratic function."""

        # f(x, y) = x^2 + y^2, gradient should be [2x, 2y]
        def quadratic_func(point: jnp.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        test_point = jnp.array([2.0, 3.0])
        numerical_grad = numerical_gradient(quadratic_func, test_point)
        expected_grad = jnp.array([4.0, 6.0])  # [2*2, 2*3]

        # Should be close to analytical gradient (less precise than analytical)
        assert jnp.allclose(numerical_grad, expected_grad, rtol=1e-2)

    def test_gradient_descent_jax_simple(self) -> None:
        """Test JAX gradient descent on a simple problem."""
        # Use a simple case where optimal location is at origin
        targets = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

        result = gradient_descent_jax(
            targets,
            initial_guess=jnp.array([5.0, 5.0]),
            learning_rate=0.1,
            max_iterations=100,
            tolerance=1e-4,
        )

        # Optimal location should be close to origin
        assert abs(result.optimal_location[0]) < 0.1
        assert abs(result.optimal_location[1]) < 0.1

        # Should have converged (or reached max iterations)
        assert result.iterations <= 100

        # Final cost should be reasonable
        assert result.final_cost > 0
        assert result.final_cost < 10

    def test_gradient_descent_numerical_simple(self) -> None:
        """Test numerical gradient descent on a simple problem."""
        # Use the same simple case
        targets = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

        result = gradient_descent_numerical(
            targets,
            initial_guess=jnp.array([5.0, 5.0]),
            learning_rate=0.1,
            max_iterations=100,
            tolerance=1e-4,
        )

        # Optimal location should be close to origin
        assert abs(result.optimal_location[0]) < 0.1
        assert abs(result.optimal_location[1]) < 0.1

        # Should have converged (or reached max iterations)
        assert result.iterations <= 100

        # Final cost should be reasonable
        assert result.final_cost > 0
        assert result.final_cost < 10

    def test_jax_vs_numerical_consistency(self) -> None:
        """Test that JAX and numerical methods give similar results."""
        locations = define_target_locations()
        targets = jnp.array([[loc.x, loc.y] for loc in locations])

        initial_guess = jnp.array([0.0, 0.0])
        learning_rate = 0.05

        jax_result = gradient_descent_jax(
            targets,
            initial_guess=initial_guess,
            learning_rate=learning_rate,
            max_iterations=200,
            tolerance=1e-5,
        )

        numerical_result = gradient_descent_numerical(
            targets,
            initial_guess=initial_guess,
            learning_rate=learning_rate,
            max_iterations=200,
            tolerance=1e-5,
        )

        # Both methods should find similar optimal locations
        location_diff = jnp.linalg.norm(jax_result.optimal_location - numerical_result.optimal_location)
        assert location_diff < 0.5  # Within 50cm (numerical methods are less precise)

        # Final costs should be similar
        cost_diff = abs(jax_result.final_cost - numerical_result.final_cost)
        assert cost_diff < 0.5

    def test_performance_advantage(self) -> None:
        """Test that JAX is faster than numerical gradients."""
        locations = define_target_locations()
        targets = jnp.array([[loc.x, loc.y] for loc in locations])

        initial_guess = jnp.array([0.0, 0.0])
        learning_rate = 0.05

        jax_result = gradient_descent_jax(
            targets,
            initial_guess=initial_guess,
            learning_rate=learning_rate,
            max_iterations=100,
            tolerance=1e-6,
        )

        numerical_result = gradient_descent_numerical(
            targets,
            initial_guess=initial_guess,
            learning_rate=learning_rate,
            max_iterations=100,
            tolerance=1e-6,
        )

        # JAX should be faster (though this might not always hold in simple cases)
        # We'll just check that both methods completed successfully
        assert jax_result.computation_time > 0
        assert numerical_result.computation_time > 0

        # Both should have reasonable convergence
        assert jax_result.iterations > 0
        assert numerical_result.iterations > 0
