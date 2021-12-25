from kedro.pipeline import Pipeline, node

from .train_model import train_model


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                train_model,
                inputs=[
                    "features",
                    "targets",
                    "parameters",
                ],
                outputs="model",
                name="model",
            ),
        ]
    )
