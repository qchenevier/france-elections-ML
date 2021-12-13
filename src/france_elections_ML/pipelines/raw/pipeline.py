from kedro.pipeline import Pipeline, node

from .download_and_extract import (
    download_and_extract_census_raw,
    download_and_extract_commune_code_raw,
    download_and_extract_IRIS_shape_raw,
    download_and_extract_municipales_2020_t1,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                download_and_extract_census_raw,
                inputs=None,
                outputs=["census_raw@text", "census_metadata_raw@text"],
                name="census_raw",
            ),
            node(
                download_and_extract_commune_code_raw,
                inputs=None,
                outputs="commune_code_raw@text",
                name="commune_code_raw",
            ),
            node(
                download_and_extract_IRIS_shape_raw,
                inputs=None,
                outputs="IRIS_shape_raw",
                name="IRIS_shape_raw",
            ),
            node(
                download_and_extract_municipales_2020_t1,
                inputs=None,
                outputs="municipales_2020_t1_raw",
                name="municipales_2020_t1_raw",
            ),
        ]
    )
