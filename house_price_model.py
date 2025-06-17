"""
House Price Model for Big City

This module implements a mathematical function that models house prices based on:
1. Location (x, y coordinates)
2. Floor number
3. Distance from high-end areas
4. Smooth price decay from premium locations
"""

import numpy as np
import plotly.graph_objects as go


def house_price_model(x, y, floor, city_params=None):
    """
    Calculate house prices based on location and floor number.

    Parameters:
    -----------
    x : float or array-like
        X coordinate(s) in the city (e.g., kilometers from reference point)
    y : float or array-like
        Y coordinate(s) in the city (e.g., kilometers from reference point)
    floor : int or array-like
        Floor number (1-based, where 1 is ground floor)
    city_params : dict, optional
        Parameters defining the city structure. If None, uses default parameters.

    Returns:
    --------
    price : float or array-like
        House price in thousands of dollars
    """

    # Default city parameters
    if city_params is None:
        city_params = {
            "high_end_centers": [
                {
                    "x": 0,
                    "y": 0,
                    "peak_price": 2000,
                    "influence_radius": 2,
                },  # City center
                {
                    "x": 2,
                    "y": 1,
                    "peak_price": 1500,
                    "influence_radius": 2.5,
                },  # Financial district
                {
                    "x": -1,
                    "y": 3,
                    "peak_price": 1200,
                    "influence_radius": 1.5,
                },  # Cultural district
                {
                    "x": 1,
                    "y": -2,
                    "peak_price": 1800,
                    "influence_radius": 2.5,
                },  # Business district
            ],
            "base_price": 500,  # Base price for remote areas
            "floor_premium": 0.02,  # 5% price increase per floor
            "distance_decay": 0.3,  # How quickly prices decay with distance
            "noise_factor": 0.001,  # Random variation factor
        }

        # Convert inputs to numpy arrays for vectorized operations
    x = np.asarray(x)
    y = np.asarray(y)
    floor = np.asarray(floor)

    # Determine the output shape by broadcasting all inputs together
    # Use numpy's broadcast function to determine the correct shape
    broadcast_result = np.broadcast(x, y, floor)
    broadcast_shape = broadcast_result.shape

    # Initialize price with base price
    price = np.full(broadcast_shape, city_params["base_price"], dtype=float)

    # Calculate influence from each high-end center and take maximum
    for center in city_params["high_end_centers"]:
        # Calculate distance from this center
        distance = np.sqrt((x - center["x"]) ** 2 + (y - center["y"]) ** 2)

        # Calculate price influence using Gaussian-like decay
        influence = center["peak_price"] * np.exp(
            -city_params["distance_decay"]
            * (distance / center["influence_radius"]) ** 2
        )

        # Take maximum between current price and this center's influence
        price = np.maximum(price, influence)

    # Apply floor premium (compound growth)
    floor_multiplier = (1 + city_params["floor_premium"]) ** (floor - 1)
    price *= floor_multiplier

    # Add some realistic noise/variation
    if city_params["noise_factor"] > 0:
        # Use deterministic noise based on coordinates for reproducibility
        np.random.seed(int(np.sum(x * 1000 + y * 1000) % 2**32))
        noise = 1 + city_params["noise_factor"] * (
            np.random.random(broadcast_shape) - 0.5
        )
        price *= noise

    return price


def visualize_price_surface(
    city_params=None, floor=1, x_range=(-8, 8), y_range=(-8, 8), resolution=100
):
    """
    Create a 3D visualization of the price surface for a given floor.

    Parameters:
    -----------
    city_params : dict, optional
        City parameters (uses default if None)
    floor : int
        Floor number to visualize
    x_range : tuple
        (min_x, max_x) range for visualization
    y_range : tuple
        (min_y, max_y) range for visualization
    resolution : int
        Number of points in each dimension

    Returns:
    --------
    fig : plotly figure
        Interactive 3D surface plot
    """

    # Create coordinate grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate prices for the entire grid
    Z = house_price_model(X, Y, floor, city_params)

    # Create 3D surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=X, y=Y, z=Z, colorscale="Viridis", colorbar=dict(title="Price (k$)")
            )
        ]
    )

    # Add markers for high-end centers
    if city_params is None:
        city_params = {
            "high_end_centers": [
                {"x": 0, "y": 0, "peak_price": 2000, "influence_radius": 5},
                {"x": 2, "y": 1, "peak_price": 1500, "influence_radius": 3},
                {"x": -1, "y": 3, "peak_price": 1200, "influence_radius": 2.5},
                {"x": 1, "y": -2, "peak_price": 1000, "influence_radius": 2},
            ],
            "base_price": 200,
            "floor_premium": 0.05,
            "distance_decay": 0.3,
            "noise_factor": 0.1,
        }

    center_x = [c["x"] for c in city_params["high_end_centers"]]
    center_y = [c["y"] for c in city_params["high_end_centers"]]
    center_z = [
        house_price_model(c["x"], c["y"], floor, city_params)
        for c in city_params["high_end_centers"]
    ]

    fig.add_trace(
        go.Scatter3d(
            x=center_x,
            y=center_y,
            z=center_z,
            mode="markers",
            marker=dict(size=10, color="red"),
            name="High-end Centers",
        )
    )

    fig.update_layout(
        title=f"House Prices - Floor {floor}",
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Price (k$)"
        ),
        width=800,
        height=600,
    )

    return fig


def visualize_price_heatmap(
    city_params=None, floor=1, x_range=(-8, 8), y_range=(-8, 8), resolution=100
):
    """
    Create a 2D heatmap visualization of prices.

    Parameters:
    -----------
    city_params : dict, optional
        City parameters (uses default if None)
    floor : int
        Floor number to visualize
    x_range : tuple
        (min_x, max_x) range for visualization
    y_range : tuple
        (min_y, max_y) range for visualization
    resolution : int
        Number of points in each dimension

    Returns:
    --------
    fig : plotly figure
        2D heatmap
    """

    # Create coordinate grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate prices for the entire grid
    Z = house_price_model(X, Y, floor, city_params)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            x=x, y=y, z=Z, colorscale="Viridis", colorbar=dict(title="Price (k$)")
        )
    )

    # Add markers for high-end centers
    if city_params is None:
        city_params = {
            "high_end_centers": [
                {"x": 0, "y": 0, "peak_price": 2000, "influence_radius": 5},
                {"x": 2, "y": 1, "peak_price": 1500, "influence_radius": 3},
                {"x": -1, "y": 3, "peak_price": 1200, "influence_radius": 2.5},
                {"x": 1, "y": -2, "peak_price": 1000, "influence_radius": 2},
            ],
            "base_price": 200,
            "floor_premium": 0.05,
            "distance_decay": 0.3,
            "noise_factor": 0.1,
        }

    center_x = [c["x"] for c in city_params["high_end_centers"]]
    center_y = [c["y"] for c in city_params["high_end_centers"]]

    fig.add_trace(
        go.Scatter(
            x=center_x,
            y=center_y,
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="High-end Centers",
        )
    )

    fig.update_layout(
        title=f"House Price Heatmap - Floor {floor}",
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        width=800,
        height=600,
    )

    return fig


def analyze_price_by_distance_and_floor():
    """
    Analyze how price varies with distance from city center and floor number.

    Returns:
    --------
    tuple : (distances, floors, prices)
        Arrays for analysis
    """

    # Analyze price vs distance from city center (0, 0)
    distances = np.linspace(0, 10, 50)
    floors = np.arange(1, 21)  # 1st to 20th floor

    # Create mesh for analysis
    D, F = np.meshgrid(distances, floors)

    # Calculate prices (along x-axis from city center)
    prices = house_price_model(D, 0, F)

    return D, F, prices


# Example usage and testing
if __name__ == "__main__":
    # Test the function with some examples
    print("House Price Model Examples:")
    print("=" * 40)

    # Example 1: City center, ground floor
    price1 = house_price_model(0, 0, 1)
    print(f"City center (0,0), Floor 1: ${price1:.0f}k")

    # Example 2: City center, 10th floor
    price2 = house_price_model(0, 0, 10)
    print(f"City center (0,0), Floor 10: ${price2:.0f}k")

    # Example 3: Suburban area
    price3 = house_price_model(5, 5, 1)
    print(f"Suburban (5,5), Floor 1: ${price3:.0f}k")

    # Example 4: Near financial district
    price4 = house_price_model(2, 1, 5)
    print(f"Near financial district (2,1), Floor 5: ${price4:.0f}k")

    # Example 5: Batch calculation
    x_coords = np.array([0, 2, -1, 5, -3])
    y_coords = np.array([0, 1, 3, -2, 4])
    floors = np.array([1, 5, 3, 1, 7])

    prices = house_price_model(x_coords, y_coords, floors)
    print("\nBatch calculation:")
    for i in range(len(x_coords)):
        print(f"  ({x_coords[i]}, {y_coords[i]}), Floor {floors[i]}: ${prices[i]:.0f}k")
