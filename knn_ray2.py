# knn_ray + object store

from knn_ray import compute_prices, load_data_points, create_query_points
import polars as pl
import ray


if __name__ == "__main__":
    ray.init()
    
    data_points = load_data_points()
    data_points_ref = ray.put(data_points)  # Store this

    query_points = create_query_points()

    # Compute prices for the query points
    prices = ray.get(compute_prices.remote(query_points, data_points_ref))
    output_df = pl.DataFrame(
        {
            "x": query_points[0],
            "y": query_points[1],
            "floor": query_points[2],
            "price": prices,
        }
    )
    print(output_df)