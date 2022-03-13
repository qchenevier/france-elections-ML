from kedro.pipeline import Pipeline, node

from .compute_targets import compute_targets
from .compute_features import compute_features


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                compute_targets,
                inputs=["municipales_2020_t1", "census_tract_code"],
                outputs="targets",
                name="targets",
            )
        ]
        + [
            node(
                compute_features,
                inputs=[
                    "census",
                    "census_tract_shape",
                    f"params:features_{features}",
                ],
                outputs=f"features_{features}",
                name=f"features_{features}",
            )
            for features in ["zero", "minimal", "light", "complex", "full"]
        ]
    )
