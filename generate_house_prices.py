import click
import json
import numpy as np
from pathlib import Path
from typing import Optional
import polars as pl

from house_price_model import house_price_model, load_city_params

SEED = 42
DEFAULT_DATA_POINTS = 10000
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "data.parquet"
DEFAULT_MAX_FLOOR = 20
DEFAULT_XY_RANGE = 10

@click.command()
@click.option("-c", "--city-definition-path", type=click.Path(path_type=Path, exists=True))
@click.option("-n", "--number-of-data-points", type=int, default=DEFAULT_DATA_POINTS)
@click.option("-o", "--output-path", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_PATH)
@click.option("-f", "--max-floor", type=int, default=DEFAULT_MAX_FLOOR)
def run(*, number_of_data_points: int, city_definition_path: Optional[Path], output_path: Path, max_floor: int):
    """
    Create a file with training data.
    """
    city_params = load_city_params(city_definition_path) if city_definition_path else None
    np.random.seed(SEED)


    x = np.random.random(number_of_data_points) * 2 * DEFAULT_XY_RANGE - DEFAULT_XY_RANGE
    y = np.random.random(number_of_data_points) * 2 * DEFAULT_XY_RANGE - DEFAULT_XY_RANGE
    floor = np.random.randint(0, max_floor + 1, number_of_data_points)

    prices = house_price_model(x, y, floor, city_params=city_params)
    df = pl.DataFrame({
        "x": x,
        "y": y,
        "floor": floor,
        "price": prices,
    })
    df.write_parquet(output_path)


if __name__ == "__main__":
    run()
