from kedro.pipeline import Pipeline, node

from .train_model import train_model
from .generate_params_and_catalogs import params_dict


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                train_model,
                inputs=[
                    f"features_{params['features']}",
                    "targets",
                    f"params:{model_name}",
                ],
                outputs=model_name,
                name=model_name,
            )
            for model_name, params in params_dict.items()
        ],
        tags="models",
    )
