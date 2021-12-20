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
                    "params:features",
                    "params:model",
                ],
                outputs="finished_training",
                name="model",
            ),
        ]
    )
