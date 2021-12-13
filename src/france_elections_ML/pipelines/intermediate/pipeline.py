from kedro.pipeline import Pipeline, node

from .compute_municipales_2020_t1 import (
    compute_municipales_2020_t1_raw_fixed,
    compute_municipales_2020_t1,
)
from .compute_census_metadata import compute_census_metadata
from .compute_census import compute_census
from .compute_census_tract_shape import (
    compute_commune_shape,
    compute_census_tract_shape,
)
from .compute_census_tract_code import (
    compute_commune_code_in_census,
    compute_commune_code,
    compute_census_tract_code,
)


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
            node(
                compute_commune_code_in_census,
                inputs="census",
                outputs="commune_code_in_census",
                name="commune_code_in_census",
            ),
            node(
                compute_commune_code,
                inputs=["commune_code_raw@CSV", "commune_code_in_census"],
                outputs="commune_code",
                name="commune_code",
            ),
            node(
                compute_census_tract_code,
                inputs=["commune_code", "commune_code_in_census"],
                outputs="census_tract_code",
                name="census_tract_code",
            ),
            node(
                compute_commune_shape,
                inputs="IRIS_shape_raw",
                outputs="commune_shape",
                name="commune_shape",
            ),
            node(
                compute_census_tract_shape,
                inputs=["commune_shape", "census_tract_code"],
                outputs="census_tract_shape",
                name="census_tract_shape",
            ),
        ]
    )
