"""
Location Optimisation Problem using JAX Autograd and Gradient Descent.

This module demonstrates how to find the optimal location that minimises
the total distance to multiple target locations (school, work, parents,
sports club) using JAX's automatic differentiation capabilities.
"""

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from knn_jax import euclidean_distances


class Location(NamedTuple):
    """Represent a 2D location with x, y coordinates."""

    x: float
    y: float
    name: str


class OptimisationResult(NamedTuple):
    """Results from the optimisation process."""

    optimal_location: jnp.ndarray
    final_cost: float
    iterations: int
    convergence_history: list[float]
    computation_time: float


def define_target_locations() -> list[Location]:
    """
    Define the four target locations for the optimisation problem.

    Returns:
    --------
    list[Location]
        List of target locations with their coordinates and names
    """
    locations = [
        Location(x=2.5, y=3.8, name="School"),
        Location(x=-1.2, y=0.5, name="Work"),
        Location(x=4.1, y=-2.3, name="Parents"),
        Location(x=-0.8, y=2.9, name="Sports Club"),
    ]
    return locations


@jax.jit
def total_distance_objective(candidate_location: jnp.ndarray, target_locations: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the total distance from a candidate location to all target locations.

    This is the objective function we want to minimise. It represents the
    sum of Euclidean distances from the candidate location to each target.
    Uses the optimised euclidean_distances function from knn_jax module.

    Parameters:
    -----------
    candidate_location : jnp.ndarray
        The candidate location [x, y] to evaluate
    target_locations : jnp.ndarray
        Array of target locations, shape (n_locations, 2)

    Returns:
    --------
    jnp.ndarray
        Total distance to all target locations (scalar array)
    """
    # Reshape candidate_location to be 2D for batch distance calculation
    candidate_reshaped = candidate_location.reshape(1, -1)

    # Calculate distances using the optimised function from knn_jax
    # This returns shape (n_targets, 1) so we need to squeeze and sum
    distances: jnp.ndarray = euclidean_distances(candidate_reshaped, target_locations)
    return jnp.sum(distances.squeeze())


def numerical_gradient(
    func: Callable[[jnp.ndarray], float], x: jnp.ndarray, epsilon: float = 1e-5
) -> jnp.ndarray:
    """
    Compute numerical gradient using finite differences.

    This function demonstrates the traditional approach to computing gradients
    numerically, which we'll compare against JAX's automatic differentiation.

    Parameters:
    -----------
    func : Callable
        Function to compute gradient for
    x : jnp.ndarray
        Point at which to compute gradient
    epsilon : float
        Small perturbation for finite differences

    Returns:
    --------
    jnp.ndarray
        Numerical gradient approximation
    """
    grad = jnp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(epsilon)
        x_minus = x.at[i].add(-epsilon)
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        grad = grad.at[i].set((f_plus - f_minus) / (2 * epsilon))
    return grad


def gradient_descent(
    target_locations: jnp.ndarray,
    grad_func: Callable[[jnp.ndarray], jnp.ndarray],
    initial_guess: jnp.ndarray = jnp.array([0.0, 0.0]),
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> OptimisationResult:
    """
    Solve the location optimisation problem using gradient descent.

    This unified function works with any gradient computation method by accepting
    the gradient function as a parameter. This allows for easy comparison between
    different gradient computation approaches (JAX autograd vs numerical).

    Parameters:
    -----------
    target_locations : jnp.ndarray
        Array of target locations, shape (n_locations, 2)
    grad_func : Callable[[jnp.ndarray], jnp.ndarray]
        Function that computes the gradient of the objective function
    initial_guess : jnp.ndarray
        Starting point for optimisation [x, y]
    learning_rate : float
        Step size for gradient descent
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for gradient norm

    Returns:
    --------
    OptimisationResult
        Complete results including optimal location and convergence history
    """
    start_time = time.perf_counter()

    # Create the objective function with target locations fixed
    objective_func = lambda x: total_distance_objective(x, target_locations)

    # Initialise optimisation variables
    current_location = initial_guess
    convergence_history = []

    for iteration in range(max_iterations):
        # Compute objective value and gradient
        current_cost = objective_func(current_location)
        gradient = grad_func(current_location)

        convergence_history.append(float(current_cost))

        gradient_norm = jnp.linalg.norm(gradient)

        # Update location using gradient descent
        current_location = current_location - learning_rate * gradient

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {float(current_cost):.4f}, Gradient norm = {gradient_norm:.2e}")

    end_time = time.perf_counter()
    computation_time = end_time - start_time

    final_cost = objective_func(current_location)

    return OptimisationResult(
        optimal_location=current_location,
        final_cost=float(final_cost),
        iterations=iteration + 1,
        convergence_history=convergence_history,
        computation_time=computation_time,
    )


def gradient_descent_jax(
    target_locations: jnp.ndarray,
    initial_guess: jnp.ndarray = jnp.array([0.0, 0.0]),
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> OptimisationResult:
    """
    Solve the location optimisation problem using JAX autograd and gradient descent.

    This function is a convenience wrapper around the generic gradient_descent function
    that uses JAX's automatic differentiation to compute exact gradients efficiently.

    Parameters:
    -----------
    target_locations : jnp.ndarray
        Array of target locations, shape (n_locations, 2)
    initial_guess : jnp.ndarray
        Starting point for optimisation [x, y]
    learning_rate : float
        Step size for gradient descent
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for gradient norm

    Returns:
    --------
    OptimisationResult
        Complete results including optimal location and convergence history
    """
    # Create the objective function with target locations fixed
    objective_func = lambda x: total_distance_objective(x, target_locations)

    # Use JAX to compute the gradient function automatically
    grad_func = jax.grad(objective_func)

    return gradient_descent(
        target_locations=target_locations,
        grad_func=grad_func,
        initial_guess=initial_guess,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def gradient_descent_numerical(
    target_locations: jnp.ndarray,
    initial_guess: jnp.ndarray = jnp.array([0.0, 0.0]),
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> OptimisationResult:
    """
    Solve the location optimisation problem using numerical gradients.

    This function is a convenience wrapper around the generic gradient_descent function
    that uses numerical differentiation to approximate gradients. It provides a
    comparison baseline against JAX's automatic differentiation.

    Parameters:
    -----------
    target_locations : jnp.ndarray
        Array of target locations, shape (n_locations, 2)
    initial_guess : jnp.ndarray
        Starting point for optimisation [x, y]
    learning_rate : float
        Step size for gradient descent
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for gradient norm

    Returns:
    --------
    OptimisationResult
        Complete results including optimal location and convergence history
    """
    # Create the objective function with target locations fixed
    objective_func = lambda x: total_distance_objective(x, target_locations)

    # Create numerical gradient function
    grad_func = lambda x: numerical_gradient(objective_func, x)

    return gradient_descent(
        target_locations=target_locations,
        grad_func=grad_func,
        initial_guess=initial_guess,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def visualise_optimisation(
    target_locations: list[Location], jax_result: OptimisationResult, numerical_result: OptimisationResult
) -> tuple[go.Figure, go.Figure]:
    """
    Create visualisations of the optimisation results.

    Parameters:
    -----------
    target_locations : list[Location]
        List of target locations
    jax_result : OptimisationResult
        Results from JAX optimisation
    numerical_result : OptimisationResult
        Results from numerical optimisation

    Returns:
    --------
    tuple[go.Figure, go.Figure]
        Location plot and convergence plot
    """
    # Extract coordinates for plotting
    target_x = [loc.x for loc in target_locations]
    target_y = [loc.y for loc in target_locations]
    target_names = [loc.name for loc in target_locations]

    # Create location visualisation
    fig_location = go.Figure()

    # Add target locations
    fig_location.add_trace(
        go.Scatter(
            x=target_x,
            y=target_y,
            mode="markers+text",
            marker=dict(size=15, color="red", symbol="star"),
            text=target_names,
            textposition="top center",
            name="Target Locations",
            textfont=dict(size=12),
        )
    )

    # Add optimal locations
    fig_location.add_trace(
        go.Scatter(
            x=[float(jax_result.optimal_location[0])],
            y=[float(jax_result.optimal_location[1])],
            mode="markers+text",
            marker=dict(size=12, color="blue", symbol="diamond"),
            text=["JAX Optimal"],
            textposition="bottom center",
            name="JAX Optimal Location",
        )
    )

    fig_location.add_trace(
        go.Scatter(
            x=[float(numerical_result.optimal_location[0])],
            y=[float(numerical_result.optimal_location[1])],
            mode="markers+text",
            marker=dict(size=12, color="green", symbol="circle"),
            text=["Numerical Optimal"],
            textposition="bottom center",
            name="Numerical Optimal Location",
        )
    )

    # Add distance lines from optimal location to targets (JAX)
    for i, loc in enumerate(target_locations):
        fig_location.add_trace(
            go.Scatter(
                x=[float(jax_result.optimal_location[0]), loc.x],
                y=[float(jax_result.optimal_location[1]), loc.y],
                mode="lines",
                line=dict(color="blue", width=1, dash="dot"),
                showlegend=False,
                opacity=0.6,
            )
        )

    fig_location.update_layout(
        title="Location Optimisation Results",
        xaxis_title="X Coordinate (km)",
        yaxis_title="Y Coordinate (km)",
        showlegend=True,
        width=800,
        height=600,
    )

    # Create convergence comparison plot
    fig_convergence = go.Figure()

    iterations_jax = list(range(len(jax_result.convergence_history)))
    iterations_numerical = list(range(len(numerical_result.convergence_history)))

    fig_convergence.add_trace(
        go.Scatter(
            x=iterations_jax,
            y=jax_result.convergence_history,
            mode="lines",
            name="JAX Autograd",
            line=dict(color="blue", width=2),
        )
    )

    fig_convergence.add_trace(
        go.Scatter(
            x=iterations_numerical,
            y=numerical_result.convergence_history,
            mode="lines",
            name="Numerical Gradients",
            line=dict(color="green", width=2),
        )
    )

    fig_convergence.update_layout(
        title="Convergence Comparison: JAX vs Numerical Gradients",
        xaxis_title="Iteration",
        yaxis_title="Total Distance (km)",
        showlegend=True,
        width=800,
        height=500,
    )

    return fig_location, fig_convergence


def main() -> None:
    """Main function to run the location optimisation demonstration."""
    print("🎯 Location Optimisation using JAX Autograd")
    print("=" * 50)

    # Step 1: Define target locations
    print("\n1️⃣ Defining target locations...")
    locations = define_target_locations()
    target_array = jnp.array([[loc.x, loc.y] for loc in locations])

    for loc in locations:
        print(f"   📍 {loc.name}: ({loc.x:.1f}, {loc.y:.1f})")

    # Step 2: Solve using JAX autograd
    print("\n2️⃣ Solving with JAX automatic differentiation...")
    jax_result = gradient_descent_jax(target_array, learning_rate=0.05)

    print("   ✨ JAX Results:")
    print(
        f"      Optimal location: ({jax_result.optimal_location[0]:.3f}, {jax_result.optimal_location[1]:.3f})"
    )
    print(f"      Final total distance: {jax_result.final_cost:.3f} km")
    print(f"      Iterations: {jax_result.iterations}")
    print(f"      Computation time: {jax_result.computation_time:.4f} seconds")

    # Step 3: Solve using numerical gradients for comparison
    print("\n3️⃣ Solving with numerical gradients...")
    numerical_result = gradient_descent_numerical(target_array, learning_rate=0.05)

    print("   📊 Numerical Results:")
    print(
        f"      Optimal location: ({numerical_result.optimal_location[0]:.3f}, {numerical_result.optimal_location[1]:.3f})"
    )
    print(f"      Final total distance: {numerical_result.final_cost:.3f} km")
    print(f"      Iterations: {numerical_result.iterations}")
    print(f"      Computation time: {numerical_result.computation_time:.4f} seconds")

    # Step 4: Performance comparison
    print("\n4️⃣ Performance Comparison:")
    speedup = numerical_result.computation_time / jax_result.computation_time
    print(f"   ⚡ JAX is {speedup:.2f}x faster than numerical gradients")
    print(
        f"   🎯 Location difference: {jnp.linalg.norm(jax_result.optimal_location - numerical_result.optimal_location):.6f} km"
    )

    # Step 5: Create visualisations
    print("\n5️⃣ Creating visualisations...")
    fig_location, fig_convergence = visualise_optimisation(locations, jax_result, numerical_result)

    # Save plots
    fig_location.write_html("location_optimisation_map.html")
    fig_convergence.write_html("location_optimisation_convergence.html")

    print("   📊 Visualisations saved:")
    print("      - location_optimisation_map.html")
    print("      - location_optimisation_convergence.html")

    print("\n✅ Location optimisation demonstration completed!")


if __name__ == "__main__":
    main()
