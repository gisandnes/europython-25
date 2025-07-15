"""
Demo script for generating scattered house price data and creating
a refined mesh using KNN regression with visualization.
"""

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from house_price_model import house_price_model
from this_tutorial.knn_jax import KNNRegressor


def generate_scattered_data(
    n_points: int = 500,
    x_range: tuple[float, float] = (-6, 6),
    y_range: tuple[float, float] = (-6, 6),
    floor_range: tuple[int, int] = (1, 10),
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate scattered house price demo data using the house price model.

    Parameters:
    -----------
    n_points : int
        Number of scattered data points to generate
    x_range : tuple
        Range for x coordinates (min, max)
    y_range : tuple
        Range for y coordinates (min, max)
    floor_range : tuple
        Range for floor numbers (min, max)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (n_points, 3) with [x, y, floor]
    y : np.ndarray
        Target prices of shape (n_points,)
    """
    np.random.seed(random_seed)

    # Generate random coordinates and floors
    x_coords = np.random.uniform(x_range[0], x_range[1], n_points)
    y_coords = np.random.uniform(y_range[0], y_range[1], n_points)
    floors = np.random.randint(floor_range[0], floor_range[1] + 1, n_points)

    # Calculate prices using the house price model
    prices = house_price_model(x_coords, y_coords, floors)

    # Combine features into matrix
    X = np.column_stack([x_coords, y_coords, floors])

    return X, prices


def create_mesh_grid(
    x_range: tuple[float, float] = (-6, 6),
    y_range: tuple[float, float] = (-6, 6),
    resolution: int = 50,
    floor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a rectangular mesh grid for prediction.

    Parameters:
    -----------
    x_range : tuple
        Range for x coordinates
    y_range : tuple
        Range for y coordinates
    resolution : int
        Number of points per dimension
    floor : int
        Fixed floor number for the mesh

    Returns:
    --------
    X_mesh : np.ndarray
        Mesh coordinates of shape (resolution^2, 3)
    x_grid : np.ndarray
        X coordinates for plotting
    y_grid : np.ndarray
        Y coordinates for plotting
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Flatten the grids and add floor information
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    floors_flat = np.full_like(x_flat, floor)

    X_mesh = np.column_stack([x_flat, y_flat, floors_flat])

    return X_mesh, x_grid, y_grid


def train_knn_and_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_mesh: np.ndarray, k: int = 15
) -> np.ndarray:
    """
    Train KNN regressor and predict prices on mesh.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_mesh : np.ndarray
        Mesh points for prediction
    k : int
        Number of neighbors for KNN

    Returns:
    --------
    y_pred : np.ndarray
        Predicted prices on mesh
    """
    # Initialize and train KNN regressor
    knn = KNNRegressor(k=k)
    knn.fit(X_train, y_train)

    # Predict on mesh
    y_pred = knn.predict(X_mesh)

    return y_pred


def visualize_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_pred: np.ndarray,
    floor: int = 5,
) -> tuple[Any, Any, Any]:
    """
    Create visualizations of the results.

    Parameters:
    -----------
    X_train : np.ndarray
        Original training data features
    y_train : np.ndarray
        Original training data targets
    x_grid : np.ndarray
        X coordinates of mesh
    y_grid : np.ndarray
        Y coordinates of mesh
    z_pred : np.ndarray
        Predicted prices on mesh
    floor : int
        Floor number used for mesh

    Returns:
    --------
    fig_heatmap : plotly figure
        2D heatmap of predicted prices
    fig_3d : plotly figure
        3D surface plot
    fig_scatter : plotly figure
        Scatter plot of original training data
    """
    # Reshape predictions to match grid
    z_grid = z_pred.reshape(x_grid.shape)

    # 1. Create heatmap visualization
    fig_heatmap = px.imshow(
        z_grid,
        x=x_grid[0, :],
        y=y_grid[:, 0],
        color_continuous_scale="Viridis",
        title=f"House Price Heatmap (Floor {floor}) - KNN Regression",
        labels={"color": "Price (k$)", "x": "X (km)", "y": "Y (km)"},
    )

    # Add scatter points of original data (same floor only)
    same_floor_mask = X_train[:, 2] == floor
    if np.any(same_floor_mask):
        fig_heatmap.add_scatter(
            x=X_train[same_floor_mask, 0],
            y=X_train[same_floor_mask, 1],
            mode="markers",
            marker=dict(color="red", size=4, opacity=0.7),
            name=f"Training Data (Floor {floor})",
        )

    # 2. Create 3D surface plot
    fig_3d = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale="Viridis",
                colorbar=dict(title="Price (k$)"),
            )
        ]
    )

    fig_3d.update_layout(
        title=f"3D House Price Surface (Floor {floor}) - KNN Regression",
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Price (k$)"),
        width=800,
        height=600,
    )

    # 3. Create scatter plot of original training data
    fig_scatter = px.scatter_3d(
        x=X_train[:, 0],
        y=X_train[:, 1],
        z=y_train,
        color=X_train[:, 2],
        title="Original Scattered Training Data",
        labels={"x": "X (km)", "y": "Y (km)", "z": "Price (k$)", "color": "Floor"},
        color_continuous_scale="Plasma",
    )

    return fig_heatmap, fig_3d, fig_scatter


def main() -> None:
    """Main function to run the complete demo."""
    print("🏠 House Price KNN Regression Demo")
    print("=" * 40)

    # Step 1: Generate scattered training data
    print("1️⃣ Generating scattered house price data...")
    X_train, y_train = generate_scattered_data(n_points=5_000, random_seed=42)
    print(f"   Generated {len(X_train)} training samples")
    print(f"   Price range: ${y_train.min():.0f}k - ${y_train.max():.0f}k")

    # Step 2: Create mesh for refined prediction
    print("\n2️⃣ Creating rectangular mesh...")
    floor_for_mesh = 5
    X_mesh, x_grid, y_grid = create_mesh_grid(resolution=80, floor=floor_for_mesh)
    print(f"   Created {len(X_mesh)} mesh points for floor {floor_for_mesh}")

    # Step 3: Train KNN and predict on mesh
    print("\n3️⃣ Training KNN regressor and predicting on mesh...")
    k_neighbors = 5
    y_pred = train_knn_and_predict(X_train, y_train, X_mesh, k=k_neighbors)
    print(f"   Used k={k_neighbors} neighbors")
    print(f"   Predicted price range: ${y_pred.min():.0f}k - ${y_pred.max():.0f}k")

    # Step 4: Create visualizations
    print("\n4️⃣ Creating visualizations...")
    fig_heatmap, fig_3d, fig_scatter = visualize_results(
        X_train, y_train, x_grid, y_grid, y_pred, floor=floor_for_mesh
    )

    # Save plots to HTML files
    print("   Saving heatmap to HTML...")
    fig_heatmap.write_html("house_prices_heatmap.html")

    print("   Saving 3D surface to HTML...")
    fig_3d.write_html("house_prices_3d_surface.html")

    print("   Saving original training data to HTML...")
    fig_scatter.write_html("house_prices_scatter.html")

    print("\n✅ Demo completed successfully!")
    print("📁 HTML files saved:")
    print("   - house_prices_heatmap.html")
    print("   - house_prices_3d_surface.html")
    print("   - house_prices_scatter.html")


def penalty(distance_km: float, alpha: float = np.log(2) / 15) -> float:
    """
    Compute the commute penalty based on distance (in km),
    assuming a travel speed of 2 km/h (i.e., 30 minutes per km).

    f(t) = max(0, exp(alpha * (t - 15)) - 1)
      where t is travel time in minutes,
      and alpha = ln(2) / 15 so that f(30) = 1.

    Parameters
    ----------
    distance_km : float
        One-way commute distance in kilometers.
    alpha : float
        Penalty factor.

    Returns
    -------
    float
        Penalty score (≈0 for t ≤ 15 min, 1 at t = 30 min, grows exponentially thereafter).
    """
    # Convert distance to time in minutes
    time_min = distance_km * 30.0

    # Compute the exponential penalty
    return max(0.0, np.exp(alpha * (time_min - 15.0)) - 1.0)


if __name__ == "__main__":
    # Run the demo
    main()
