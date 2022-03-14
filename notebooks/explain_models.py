# %%
from pathlib import Path
import re
import copy

import pandas as pd
import polars as pl
import numpy as np
import torch
from anchor import anchor_tabular
from tqdm import tqdm

from kedro.extras.extensions.ipython import reload_kedro
from kedro.io.data_catalog import CREDENTIALS_KEY
import plotly.graph_objects as go
import plotly.express as px


# %%
def flatten_column_names(df):
    df = df.copy()
    df.columns = (
        pd.Series(df.columns.tolist())
        .apply(lambda x: [str(c) for c in x])
        .str.join("_")
        .str.strip("_")
        .tolist()
    )
    return df


def _not_regex(reg):
    return rf"^((?!{reg}).)*$"


def compute_predictions_from_raw_predictions(df, drop_raw=True):
    df = df.copy()
    for c in df.filter(regex=r"_raw$").columns:
        df[c[:-4]] = df[c] * df["population"]
    if drop_raw:
        return df.filter(regex=_not_regex(r"_raw$"))
    return df


def compute_densite_population_from_features(features):
    features_with_targets_non_null = compute_features_with_targets_non_null(
        features
    )
    densite_population = (
        (features_with_targets_non_null)
        .to_pandas()
        .groupby("code_census_tract")
        .agg({"densite_lognorm": "first", "population": "sum"})
        .reset_index()
    )
    return densite_population


def compute_metadata_from_model_name(model_name):
    metadata = dict(model_name=model_name)
    pattern = r"^model_(?P<features_set>.*)_(?P<run_number>[0-9]*)"
    metadata.update(re.match(pattern, model_name).groupdict())
    run_name = f"{metadata['features_set']}_{metadata['run_number']}"
    metadata.update({"run_name": run_name})
    features_name = f"features_{metadata['features_set']}"
    metadata.update({"features_name": features_name})
    return metadata


def compute_features_with_targets_non_null(features):
    return (features).filter(
        pl.col("code_census_tract").is_in(
            targets_non_null.code_census_tract.unique().tolist()
        )
    )


def load_run(metadata, cache, catalog):
    def _cache_or_load(item_name, cache, catalog):
        if item_name not in cache:
            cache[item_name] = catalog.load(item_name)
        return cache[item_name]

    return dict(
        model=_cache_or_load(metadata["model_name"], cache, catalog),
        features=_cache_or_load(metadata["features_name"], cache, catalog),
    )


def compute_predictions(features, model, targets):
    n_targets = targets.shape[1] - 1
    X = features[:, 2:].to_numpy()
    raw_predictions = pd.DataFrame(
        model.model(torch.tensor(X)).detach().numpy().reshape(-1, n_targets),
        columns=targets.columns[1:],
    ).add_suffix("_raw")
    predictions = (
        (features)
        .select(["code_census_tract", "population"])
        .to_pandas()
        .join(raw_predictions)
        .pipe(compute_predictions_from_raw_predictions)
        .drop(["population"], axis=1)
        .groupby("code_census_tract")
        .sum()
        .round()
        .astype(int)
        .reset_index()
    )
    return predictions


def compute_predictions_from_runs(runs, targets):
    def _add_suffix(df, suffix, no_suffix):
        return (df).set_index(no_suffix).add_suffix(suffix).reset_index()

    key_column = "code_census_tract"
    if isinstance(runs, dict):
        run = runs
        features = run["features"]
        suffix = f"_predicted_{run['run_name']}"
        model = run["model"]
        return (
            (features)
            .pipe(compute_features_with_targets_non_null)
            .pipe(compute_predictions, model, targets)
            .pipe(_add_suffix, suffix, key_column)
        )
    return pd.concat(
        [
            compute_predictions_from_runs(run, targets).set_index(key_column)
            for run in runs
        ],
        axis=1,
    ).reset_index()


def add_predictions_from_runs(targets, runs):
    return (
        (targets)
        .set_index("code_census_tract")
        .add_suffix("_actual")
        .merge(
            compute_predictions_from_runs(runs, targets).set_index(
                "code_census_tract"
            ),
            on="code_census_tract",
            how="inner",
        )
        .reset_index()
    )


# %%
startup_path = Path.cwd()
project_path = startup_path.parent
reload_kedro(project_path)

# %%
cache = dict()
model_names = list(filter(lambda x: x.startswith("model_"), catalog.list()))
runs_metadata = [
    compute_metadata_from_model_name(model_name) for model_name in model_names
]
runs = [
    {**metadata, **load_run(metadata, cache, catalog)}
    for metadata in runs_metadata
]

# %%
targets = catalog.load("targets")
targets_non_null = targets.loc[lambda df: ~(df == 0).any(axis=1)]
census_metadata = catalog.load("census_metadata")

# %%
features = runs[0]["features"]
features_with_targets_non_null = compute_features_with_targets_non_null(
    features
)
densite_population = compute_densite_population_from_features(features)

# %%
model_selection = [
    "model_minimal_005",
    "model_minimal_007",
    "model_minimal_000",
    "model_zero_001",
    "model_zero_008",
    "model_zero_013",
]

# %%
# %%
model = runs[0]["model"]


def model_wrapped(X, rounded=True, target_idx=3):
    prediction = (
        model.model(torch.tensor(X))
        .detach()
        .numpy()
        .reshape(-1, 5)[:, target_idx]
    )
    if rounded:
        return prediction.round()
    return prediction


# %%
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=targets.columns[1 + target_idx],
    feature_names=features.columns[2:],
    train_data=X,
)


def explain(row):
    return explainer.explain_instance(
        row,
        model_wrapped,
    )


# %%
example = X[np.random.randint(X.shape[0])]
explanation = explain(example)
print(
    f"""
prediction: {model_wrapped(example, rounded=False)}
names: {[cond for cond in explanation.names() if "sexe" in cond or "<= 0.00" not in cond]}
precision: {explanation.precision()}
coverage: {explanation.coverage()}
"""
)


# %%
def compute_condition_value_and_variable(df):
    return (
        df.assign(
            condition_split=lambda df: (df.condition)
            .str.replace("=", "")
            .str.split(r"(<|>)")
        )
        .assign(
            condition_right=lambda df: (df.condition_split)
            .str[-1]
            .pipe(pd.to_numeric, errors="coerce")
        )
        .assign(
            condition_left=lambda df: (df.condition_split)
            .str[0]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(df.condition_right)
        )
        .assign(
            bonus=lambda df: (df.condition_split.str[1] == ">").astype(float)
        )
        .assign(
            value=lambda df: (df.condition_right + df.condition_left) / 2
            + df.bonus
        )
        .assign(variable=lambda df: df.condition_split.str[-3])
        .drop(
            ["condition_split", "condition_left", "condition_right", "bonus"],
            axis=1,
        )
    )


# %%
N = 100
X_sample_shuffled = np.random.permutation(X_sample)[:N]

# %%
explanations = [explain(X_sample_shuffled[i, :]) for i in tqdm(range(N))]

# %%
# predictions = [model_wrapped(X_sample_shuffled[i, :])[0] for i in tqdm(range(N))]
predictions = [
    model_wrapped(X_sample_shuffled[i, :])[0] >= 1 for i in tqdm(range(N))
]

# %%


df_explanations = (
    pd.DataFrame(
        {
            "condition": [e.names() for e in explanations],
            "prediction": predictions,
        }
    )
    .explode("condition")
    .pipe(compute_condition_value_and_variable)
    .assign(value=lambda df: df.value + (df.value == 0.5) / 2)
)

explanation_pivot = (
    (df_explanations)
    .loc[:, ["prediction", "variable", "value"]]
    .assign(
        variable=lambda df: (df.variable)
        .str.strip()
        .str.replace(r"[0-9]+$", lambda x: x.group(0).zfill(2))
    )
    .sort_values(by="variable")
    .pivot_table(
        index="variable",
        columns="prediction",
        values="value",
        aggfunc=["mean", "count"],
    )
    .reset_index()
)

value_descriptions = (
    explanation_pivot.variable.str.extract(r"((.*)_([0-9Z]+))$")
    .rename(columns={0: "variable", 1: "variable_name", 2: "value_code"})
    # .assign(value_code=lambda df: df.value_code.str.lstrip("0"))
    .merge(
        census_metadata.assign(
            value_code=lambda df: df.value_code.str.zfill(2)
        ).loc[:, ["variable_name", "value_code", "value_description"]],
        how="left",
    )
    .dropna()
    .loc[:, ["variable", "value_description"]]
)

with pd.option_context("precision", 2):
    explanation_grid = (
        explanation_pivot.pipe(flatten_column_names)
        .merge(value_descriptions, on="variable", how="left")
        .style.background_gradient()
    )

explanation_grid

# %%

# %%
census_metadata

# %%
census_metadata.loc[lambda df: df.variable_name == "statut_conjugal"]

# %%
# %%
# %%
# %%
# import altair as alt
from lime.lime_tabular import LimeTabularExplainer

# %%
# pl.utilities.seed.seed_everything(7)

# # %%
# df_features_filepath = "data/df_features.hdf5"
# df_targets_filepath = "data/df_targets.parquet"

# # %%
# df_features = vaex.open(df_features_filepath)
# df_targets = pd.read_parquet(df_targets_filepath)
# df_targets_non_null = df_targets.loc[lambda df: ~(df == 0).any(axis=1)]

# # %%
# dataset = MasterDataset(
#     df_features,
#     df_targets_non_null,
#     ID_column="code_census_tract",
# )

# # %%
# not_feature_or_target = ["code_census_tract", "population"]
# n_features = len(set(dataset.features) - set(not_feature_or_target))
# n_targets = len(set(dataset.targets) - set(not_feature_or_target))
# log_folder = "lightning_logs"
# model = AggregateModel.load_from_checkpoint(
#     get_last_checkpoint(log_folder=log_folder),
#     n_features=n_features,
#     n_targets=n_targets,
#     n_hidden_layers=1,
# )

# %%
# X = torch.tensor(dataset[0][0][:, 2:])
# X_sample = torch.unique(X, dim=0).detach().numpy()

# target_idx = 3


# def model_wrapped(X):
#     return (
#         model.model(torch.tensor(X))
#         .detach()
#         .numpy()
#         .reshape(-1, 5)[:, target_idx]
#     )


# %%
categorical_features = (
    [c for c in features.columns[2:] if re.match(r".*_[0-9Z]+$", c)],
)

explainer = LimeTabularExplainer(
    X,
    feature_names=features.columns[2:],
    categorical_features=categorical_features,
    class_names=targets.columns[target_idx + 1],
    mode="regression",
    # verbose=True,
)


def explain(row):
    return explainer.explain_instance(
        row,
        model_wrapped,
        labels=model_wrapped(row),
        num_features=row.shape[0],
        num_samples=200,
    )


def compute_condition_value_and_variable(df):
    return (
        df.assign(
            condition_split=lambda df: df.condition.str.replace(
                "=", ""
            ).str.split(r"(<|>)")
        )
        .assign(
            condition_right=lambda df: df.condition_split.str[-1].pipe(
                pd.to_numeric, errors="coerce"
            )
        )
        .assign(
            condition_left=lambda df: df.condition_split.str[0]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(df.condition_right)
        )
        .assign(
            bonus=lambda df: (df.condition_split.str[1] == ">").astype(float)
        )
        .assign(
            value=lambda df: (df.condition_right + df.condition_left) / 2
            + df.bonus
        )
        .assign(variable=lambda df: df.condition_split.str[-3])
        .drop(
            ["condition_split", "condition_left", "condition_right", "bonus"],
            axis=1,
        )
    )


# %%
# X_sample_shuffled = np.random.permutation(X_sample)
explanations = [explain(X_sample_shuffled[i]) for i in tqdm(range(100))]

# %%
df_explanations = (
    pd.DataFrame(
        [e for explanation in explanations for e in explanation.as_list()],
        columns=["condition", "impact"],
    )
    .pipe(compute_condition_value_and_variable)
    .assign(value=lambda df: df.value + (df.value == 0.5) / 2)
)


# %%
def normalize(serie):
    return (serie - serie.mean()) / serie.std()


df_plot = (
    (df_explanations)
    .loc[:, ["impact", "variable", "value"]]
    .assign(
        variable=lambda df: (df.variable)
        .str.strip()
        .str.replace(r"[0-9]+$", lambda x: x.group(0).zfill(2))
    )
    .sort_values(by="variable")
    .groupby("variable", as_index=False)
    .apply(
        lambda df: df.assign(
            value_impact=lambda df: normalize(df.value) * df.impact
        )
    )
    .reset_index(drop=True)
)

# %%
import seaborn as sns

height = 1
g = sns.FacetGrid(
    df_plot, row="variable", hue="value", height=height, aspect=8 / height
)
g.map(sns.histplot, "impact")

# %%
