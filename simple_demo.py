"""
Simple House Price Model Demonstration

Mathematical function for modeling house prices in a big city.
Pure Python implementation without external dependencies.
"""

import math


def house_price_model(x, y, floor):
    """
    Calculate house prices based on location and floor number.

    Mathematical Function:
    P(x,y,f) = [B + Σᵢ Aᵢ·exp(-α·((x-xᵢ)² + (y-yᵢ)²)/rᵢ²)] × (1+p)^(f-1)

    Parameters:
    -----------
    x : float
        X coordinate in the city (kilometers from reference point)
    y : float
        Y coordinate in the city (kilometers from reference point)
    floor : int
        Floor number (1-based, where 1 is ground floor)

    Returns:
    --------
    price : float
        House price in thousands of dollars
    """

    # City configuration
    high_end_centers = [
        {"x": 0, "y": 0, "peak_price": 2000, "influence_radius": 5},  # City center
        {
            "x": 2,
            "y": 1,
            "peak_price": 1500,
            "influence_radius": 3,
        },  # Financial district
        {
            "x": -1,
            "y": 3,
            "peak_price": 1200,
            "influence_radius": 2.5,
        },  # Cultural district
        {
            "x": 1,
            "y": -2,
            "peak_price": 1000,
            "influence_radius": 2,
        },  # Business district
    ]

    base_price = 200  # Base price for remote areas
    floor_premium = 0.05  # 5% price increase per floor
    distance_decay = 0.3  # How quickly prices decay with distance

    # Initialize price with base price
    price = base_price

    # Add influence from each high-end center
    for center in high_end_centers:
        # Calculate distance from this center
        distance = math.sqrt((x - center["x"]) ** 2 + (y - center["y"]) ** 2)

        # Calculate price influence using Gaussian-like decay
        normalized_distance = distance / center["influence_radius"]
        influence = center["peak_price"] * math.exp(
            -distance_decay * normalized_distance**2
        )

        # Add this center's influence to the total price
        price += influence

    # Apply floor premium (compound growth)
    floor_multiplier = (1 + floor_premium) ** (floor - 1)
    price *= floor_multiplier

    return price


def main():
    print("🏠 House Price Model - Mathematical Function")
    print("=" * 55)

    print("\nMathematical Formula:")
    print("P(x,y,f) = [B + Σᵢ Aᵢ·exp(-α·((x-xᵢ)² + (y-yᵢ)²)/rᵢ²)] × (1+p)^(f-1)")
    print("\nWhere:")
    print("  P(x,y,f) = Price at location (x,y) and floor f")
    print("  B        = Base price (distant areas) = 200k")
    print("  Aᵢ       = Peak price influence of center i")
    print("  (xᵢ,yᵢ)  = Coordinates of high-end center i")
    print("  rᵢ       = Influence radius of center i")
    print("  α        = Distance decay factor = 0.3")
    print("  p        = Floor premium rate = 5%")
    print("  f        = Floor number")

    print("\nHigh-End Centers:")
    centers = [
        "City Center (0,0): $2000k peak, 5km radius",
        "Financial District (2,1): $1500k peak, 3km radius",
        "Cultural District (-1,3): $1200k peak, 2.5km radius",
        "Business District (1,-2): $1000k peak, 2km radius",
    ]
    for center in centers:
        print(f"  • {center}")

    # 1. Basic Examples
    print("\n1. Price Examples:")
    print("-" * 25)

    examples = [
        (0, 0, 1, "City Center, Ground Floor"),
        (0, 0, 10, "City Center, 10th Floor"),
        (0, 0, 20, "City Center, 20th Floor"),
        (2, 1, 1, "Financial District, Ground Floor"),
        (2, 1, 10, "Financial District, 10th Floor"),
        (-1, 3, 5, "Cultural District, 5th Floor"),
        (1, -2, 3, "Business District, 3rd Floor"),
        (5, 5, 1, "Suburban Area, Ground Floor"),
        (8, 8, 1, "Far Suburban, Ground Floor"),
    ]

    print(f"{'Location':<25} {'Floor':<6} {'Price':<8}")
    print("-" * 40)

    for x, y, floor, description in examples:
        price = house_price_model(x, y, floor)
        location = f"({x},{y})"
        print(f"{location:<25} {floor:<6} ${price:.0f}k")

    # 2. Distance Analysis
    print("\n2. Price vs Distance from City Center:")
    print("-" * 42)

    distances = [0, 1, 2, 3, 4, 5, 8, 10]

    print(f"{'Distance (km)':<15} {'Floor 1':<10} {'Floor 10':<10}")
    print("-" * 35)

    for dist in distances:
        price_1f = house_price_model(dist, 0, 1)
        price_10f = house_price_model(dist, 0, 10)
        print(f"{dist:<15} ${price_1f:<9.0f} ${price_10f:<9.0f}")

    # 3. Floor Premium Analysis
    print("\n3. Floor Premium Analysis (City Center):")
    print("-" * 40)

    floors = [1, 5, 10, 15, 20, 25, 30]

    print(f"{'Floor':<6} {'Price':<8} {'Premium vs Floor 1':<16}")
    print("-" * 30)

    base_price_1f = house_price_model(0, 0, 1)

    for floor in floors:
        price = house_price_model(0, 0, floor)
        premium = (price / base_price_1f - 1) * 100
        print(f"{floor:<6} ${price:<7.0f} {premium:>8.1f}%")

    # 4. Cross-section Analysis
    print("\n4. Price Cross-section (y=0, Floor 1):")
    print("-" * 35)

    x_values = [-6, -4, -2, -1, 0, 1, 2, 3, 4, 6]

    print(f"{'X Position':<12} {'Price':<8}")
    print("-" * 20)

    for x in x_values:
        price = house_price_model(x, 0, 1)
        print(f"{x:<12} ${price:<7.0f}")

    print("\n5. Key Properties Demonstrated:")
    print("-" * 32)
    print("✓ Multiple high-end centers with different peak prices")
    print("✓ Smooth price decay based on distance from centers")
    print("✓ Function of cartesian coordinates (x, y)")
    print("✓ Price increases with higher floors (5% compound)")
    print("✓ Realistic base price for distant areas")
    print("✓ Gaussian-like decay creates smooth transitions")


if __name__ == "__main__":
    main()
