# knn_ray + object store + batches

from knn_ray import compute_prices, load_data_points, create_query_points
import polars as pl
import numpy as np
import ray


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    ray.init()
    
    data_points = load_data_points()
    data_points_ref = ray.put(data_points)  # Store this

    query_points = create_query_points(1000)
    batch_size = 1000

    query_point_batches = [
        query_points[:,i:i+batch_size]
        for i in range(0, query_points.shape[1] // batch_size * batch_size, batch_size)
    ]
    logger.info(f"Submitting {len(query_point_batches)} batches of query points")

    # Compute prices for the query points   
    prices = ray.get([
        compute_prices.remote(query_point_batch, data_points_ref)
        for query_point_batch in query_point_batches
    ])
    prices = np.concatenate(prices)
    output_df = pl.DataFrame(
        {
            "x": query_points[0],
            "y": query_points[1],
            "floor": query_points[2],
            "price": prices,
        }
    )
    print(output_df)