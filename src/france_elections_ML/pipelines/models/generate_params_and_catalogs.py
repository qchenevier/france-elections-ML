from pathlib import Path

import yaml
from sklearn.model_selection import ParameterGrid
from unflatten import unflatten


def model_name(params, i, N):
    return f"model_{params['features']}_{i % N:03d}"


params_grid = list(
    ParameterGrid(
        {
            "features": ["minimal", "light", "complex", "full"],
            "trainer.seed": range(5),
            "trainer.max_epochs": [30],
            "model.hidden_layers": range(4),
            "model.hidden_size_factor": [0.5, 1, 2, 4],
            "model.output_activation": ["GELU"],
            "model.hidden_activation": ["GELU"],
        }
    )
)

N = int(len(params_grid) / 4)

params_dict = {
    model_name(params, i, N): unflatten(params)
    for i, params in enumerate(params_grid)
}

catalog_dict_local = {
    model_name(params, i, N): {
        "type": "pickle.PickleDataSet",
        "filepath": f"data/06_models/{model_name(params, i, N)}.pkl",
    }
    for i, params in enumerate(params_grid)
}

catalog_dict_base = {
    model_name(params, i, N): {
        "type": "pickle.PickleDataSet",
        "filepath": f"s3://france-elections-ml/06_models/{model_name(params, i, N)}.pkl",
    }
    for i, params in enumerate(params_grid)
}

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent.parent.parent
    conf = root / "conf"

    with open(conf / "base" / "parameters_model.yml", "w") as f:
        yaml.dump(params_dict, f)

    with open(conf / "base" / "catalog_model.yml", "w") as f:
        yaml.dump(catalog_dict_base, f)

    with open(conf / "local" / "catalog_model.yml", "w") as f:
        yaml.dump(catalog_dict_local, f)
