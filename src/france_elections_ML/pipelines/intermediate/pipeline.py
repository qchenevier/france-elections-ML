from kedro.pipeline import Pipeline, node

from .compute_municipales_2020_t1 import (
    compute_municipales_2020_t1_raw_fixed,
    compute_municipales_2020_t1,
)
from .compute_census_metadata import compute_census_metadata
from .compute_census import compute_census

def create_pipeline() -> Pipeline:
    pass
    return Pipeline(
        [
            node(
                compute_municipales_2020_t1_raw_fixed,
                inputs="municipales_2020_t1_raw",
                outputs="municipales_2020_t1_raw_fixed@text",
                name="municipales_2020_t1_raw_fixed",
            ),
            node(
                compute_municipales_2020_t1,
                inputs="municipales_2020_t1_raw_fixed@CSV",
                outputs="municipales_2020_t1",
                name="municipales_2020_t1",
            ),
            node(
                compute_census_metadata,
                inputs="census_metadata_raw@CSV",
                outputs="census_metadata",
                name="census_metadata",
            ),
            node(
                compute_census,
                inputs=["census_raw@CSV", "census_metadata"],
                outputs="census",
                name="census",
            ),
        ]
    )
