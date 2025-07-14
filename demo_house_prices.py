"""
Demonstration of the House Price Model

This script demonstrates the mathematical function for modeling house prices
in a big city, showing examples and creating visualizations.
"""

import numpy as np

from house_price_model import (
    house_price_model,
    visualize_price_heatmap,
    visualize_price_surface,
)


def main():
    print("🏠 House Price Model Demonstration")
    print("=" * 50)

    # 1. Basic Examples
    print("\n1. Basic Price Examples:")
    print("-" * 25)

    # City center examples
    price_center_1f = house_price_model(0, 0, 1)
    price_center_10f = house_price_model(0, 0, 10)
    price_center_20f = house_price_model(0, 0, 20)

    print("City Center (0,0):")
    print(f"  Floor 1:  ${price_center_1f:.0f}k")
    print(f"  Floor 10: ${price_center_10f:.0f}k")
    print(f"  Floor 20: ${price_center_20f:.0f}k")

    # Financial district examples
    price_financial_1f = house_price_model(2, 1, 1)
    price_financial_10f = house_price_model(2, 1, 10)

    print("\nFinancial District (2,1):")
    print(f"  Floor 1:  ${price_financial_1f:.0f}k")
    print(f"  Floor 10: ${price_financial_10f:.0f}k")

    # Suburban examples
    price_suburb_1f = house_price_model(6, 6, 1)
    price_suburb_10f = house_price_model(6, 6, 10)

    print("\nSuburban Area (6,6):")
    print(f"  Floor 1:  ${price_suburb_1f:.0f}k")
    print(f"  Floor 10: ${price_suburb_10f:.0f}k")

    # 2. Distance Analysis
    print("\n2. Price vs Distance Analysis:")
    print("-" * 30)

    distances = np.array([0, 1, 2, 3, 4, 5, 8, 10])
    prices_1f = house_price_model(distances, 0, 1)  # Along x-axis, floor 1
    prices_10f = house_price_model(distances, 0, 10)  # Along x-axis, floor 10

    print("Distance from center | Floor 1 | Floor 10")
    print("-" * 42)
    for i, dist in enumerate(distances):
        print(f"{dist:8.0f} km        | ${prices_1f[i]:6.0f}k | ${prices_10f[i]:7.0f}k")

    # 3. Floor Premium Analysis
    print("\n3. Floor Premium Analysis (City Center):")
    print("-" * 40)

    floors = np.array([1, 5, 10, 15, 20, 25, 30])
    prices_floors = house_price_model(0, 0, floors)

    print("Floor | Price  | Premium vs Floor 1")
    print("-" * 35)
    for i, floor in enumerate(floors):
        premium = (prices_floors[i] / prices_floors[0] - 1) * 100
        print(f"{floor:5d} | ${prices_floors[i]:5.0f}k | {premium:8.1f}%")

    # 4. Mathematical Function Description
    print("\n4. Mathematical Function:")
    print("-" * 25)
    print(
        "P(x,y,f) = max(B, max(Aᵢ·exp(-α·((x-xᵢ)² + (y-yᵢ)²)/rᵢ²))) × (1+p)^(f-1) × N"
    )
    print()
    print("Where:")
    print("  P(x,y,f) = Price at location (x,y) and floor f")
    print("  B        = Base price (minimum price, distant areas)")
    print("  Aᵢ       = Peak price influence of center i")
    print("  (xᵢ,yᵢ)  = Coordinates of high-end center i")
    print("  rᵢ       = Influence radius of center i")
    print("  α        = Distance decay factor")
    print("  p        = Floor premium rate")
    print("  f        = Floor number")
    print("  N        = Noise factor (random variation)")
    print("  max()    = Takes maximum influence (no additive effects)")

    # 5. Custom City Configuration
    print("\n5. Custom City Example:")
    print("-" * 25)

    # Define a custom city with different parameters
    custom_city = {
        "high_end_centers": [
            {"x": 0, "y": 0, "peak_price": 3000, "influence_radius": 4},  # Main center
            {"x": 3, "y": 2, "peak_price": 1800, "influence_radius": 2.5},  # Tech hub
            {
                "x": -2,
                "y": -1,
                "peak_price": 1500,
                "influence_radius": 2,
            },  # Arts district
        ],
        "base_price": 150,
        "floor_premium": 0.08,  # 8% premium per floor
        "distance_decay": 0.4,
        "noise_factor": 0.001,
    }

    # Compare prices in custom city
    locations = [(0, 0), (3, 2), (-2, -1), (5, 5)]

    print("Location      | Default City | Custom City")
    print("-" * 44)
    for x, y in locations:
        price_default = house_price_model(x, y, 1)
        price_custom = house_price_model(x, y, 1, custom_city)
        print(
            f"({x:2d},{y:2d})        | ${price_default:7.0f}k    | ${price_custom:6.0f}k"
        )


if __name__ == "__main__":
    main()

    # Uncomment the following lines to generate visualizations
    # (requires plotly to be installed)

    print("\n6. Generating Visualizations...")
    print("-" * 30)

    try:
        # Create and show 3D surface plot
        fig_3d = visualize_price_surface(floor=1)
        fig_3d.write_html("price_surface_3d.html")
        print("3D surface plot saved as 'price_surface_3d.html'")

        # Create and show 2D heatmap
        fig_2d = visualize_price_heatmap(floor=1)
        fig_2d.write_html("price_heatmap_2d.html")
        print("2D heatmap saved as 'price_heatmap_2d.html'")

        # Create comparison between different floors
        fig_floors = visualize_price_heatmap(floor=10)
        fig_floors.write_html("price_heatmap_floor10.html")
        print("Floor 10 heatmap saved as 'price_heatmap_floor10.html'")

        print("\nVisualizations created successfully!")
        print("Open the HTML files in your web browser to view interactive plots.")

    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        print("Visualizations skipped.")
