from pathlib import Path

import yaml
from sklearn.model_selection import ParameterGrid
from unflatten import unflatten


def model_name(params, i, N):
    return f"model_{params['features']}_seed{params['seed']}_id{i:03d}"


grid_dimensions = {
    "features": ["zero", "minimal", "light"],
    "seed": list(range(900, 900 + 6)),
    "max_epochs": [1000],
    "hidden_layers": [1],
    "hidden_size_factor": [0.5, 0.7, 0.9],
    "output_activation": ["Softplus"],
    "hidden_activation": ["GELU"],
}

grid = list(ParameterGrid(grid_dimensions))

N = int(
    len(grid) / len(grid_dimensions["features"]) / len(grid_dimensions["seed"])
)

params_dict = {
    model_name(params, i, N): dict(
        **unflatten(params), model_name=model_name(params, i, N)
    )
    for i, params in enumerate(grid)
}

catalog_dict_local = {
    model_name(params, i, N): {
        "type": "pickle.PickleDataSet",
        "filepath": f"data/06_models/{model_name(params, i, N)}.pkl",
    }
    for i, params in enumerate(grid)
}

catalog_dict_base = {
    model_name(params, i, N): {
        "type": "pickle.PickleDataSet",
        "filepath": f"s3://france-elections-ml/06_models/{model_name(params, i, N)}.pkl",
    }
    for i, params in enumerate(grid)
}

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent.parent.parent
    conf = root / "conf"

    with open(conf / "base" / "parameters_model.yml", "w") as f:
        yaml.dump(params_dict, f)

    with open(conf / "base" / "catalog_model.yml", "w") as f:
        yaml.dump(catalog_dict_base, f)
