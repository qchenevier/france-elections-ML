# %%
from pathlib import Path
import logging
import re

import pandas as pd
import polars as pl
import torch

from kedro.extras.extensions.ipython import reload_kedro
import plotly.express as px
from kedro.io import DataSetError


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
    densite_population = (
        (features)
        .to_pandas()
        .groupby("code_census_tract")
        .agg({"densite_lognorm": "first", "population": "sum"})
        .reset_index()
    )
    return densite_population


def compute_metadata_from_model_name(model_name):
    metadata = dict(model_name=model_name)
    pattern = r"^model_(?P<features>.*)_seed(?P<seed>[0-9]*)_id(?P<id>[0-9]*)"
    metadata.update(re.match(pattern, model_name).groupdict())
    run_name = (
        f"{metadata['features']}_seed{metadata['seed']}_id{metadata['id']}"
    )
    metadata.update({"run_name": run_name})
    features_name = f"features_{metadata['features']}"
    metadata.update({"features_name": features_name})
    return metadata


def compute_features_with_targets_non_null(features, targets_non_null):
    return (features).filter(
        pl.col("code_census_tract").is_in(
            targets_non_null.code_census_tract.unique().tolist()
        )
    )


def load_run(metadata, cache, catalog):
    def _cache_or_load(item_name, cache, catalog):
        if item_name not in cache:
            try:
                cache[item_name] = catalog.load(item_name)
            except DataSetError as e:
                cache[item_name] = e
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

    targets_non_null = targets.loc[lambda df: ~(df == 0).any(axis=1)]
    key_column = "code_census_tract"
    if isinstance(runs, dict):
        run = runs
        features = run["features"]
        suffix = "_" + run["model_name"]
        model = run["model"]
        logging.info("Computing predictions from run: %s", run["model_name"])
        return (
            (features)
            .pipe(compute_features_with_targets_non_null, targets_non_null)
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
        .add_suffix("_truth")
        .merge(
            compute_predictions_from_runs(runs, targets).set_index(
                "code_census_tract"
            ),
            on="code_census_tract",
            how="inner",
        )
        .reset_index()
    )


def compute_targets_and_predictions(targets, runs, model_selection=None):
    runs_selected = (
        filter(lambda r: r["model_name"] in model_selection, runs)
        if model_selection
        else runs
    )
    return targets.pipe(add_predictions_from_runs, runs_selected)


def compute_targets_and_predictions_long(targets_and_predictions):
    stubnames = (
        (targets_and_predictions)
        .filter(regex="_truth")
        .rename(columns=lambda c: c.replace("_truth", ""))
        .columns.tolist()
    )
    return (
        (targets_and_predictions)
        .pipe(
            pd.wide_to_long,
            stubnames=stubnames,
            sep="_",
            i="code_census_tract",
            j="model",
            suffix=r"\w+",
        )
        .reset_index()
        .melt(
            id_vars=["code_census_tract", "model"],
            var_name="target",
        )
    )


def compute_targets_and_predictions_per_target(targets_and_predictions_long):
    return (
        (targets_and_predictions_long)
        .pivot(
            index=["code_census_tract", "target"],
            columns=["model"],
        )
        .reset_index()
        .pipe(flatten_column_names)
        .rename(columns=lambda s: s.replace("value_", ""))
    )


def normalize_by_column(df, col):
    df = df.copy()
    for c in df.columns:
        if c != col:
            df[c] = df[c] / df[col]
    df[col] = 1
    return df


def compute_residuals_per_target(
    targets_and_predictions_per_target, reference_column="truth"
):
    return (
        (targets_and_predictions_per_target)
        .set_index(["code_census_tract", "target"])
        .pipe(normalize_by_column, reference_column)
        .reset_index()
    )


def compute_residuals_long(residuals_per_target):
    return (residuals_per_target).melt(
        id_vars=["code_census_tract", "target"],
        value_name="value_ratio",
        var_name="model",
    )


def compute_facet_params(model_selection, target_selection, plot_size=200):
    color_discrete_map = {
        "inscrits": "darkslategray",
        "voix": "slategray",
        "gauche": "red",
        "droite": "blue",
        "extreme_droite": "darkblue",
    }
    target_orders = {
        "target": target_selection,
        "model": model_selection,
    }
    width = plot_size * len(model_selection) + 135
    height = plot_size * len(target_selection) + 105
    return dict(
        template="plotly_white",
        facet_col="model",
        facet_row="target",
        color="target",
        color_discrete_map=color_discrete_map,
        category_orders=target_orders,
        width=width,
        height=height,
    )


def plot_predictions(
    targets_and_predictions_long,
    densite_population,
    model_selection,
    target_selection,
):
    facet_kwargs = compute_facet_params(model_selection, target_selection)
    fig = (
        (targets_and_predictions_long)
        .loc[lambda df: df.model.isin(model_selection)]
        .loc[lambda df: df.target.isin(target_selection)]
        .merge(densite_population, on="code_census_tract")
        .assign(value_per_capita=lambda df: df.value / df.population)
        .rename(
            columns={
                "densite_lognorm": "Density (log normalized)",
                "value_per_capita": "Prediction",
            }
        )
        .plot(
            kind="scatter",
            x="Density (log normalized)",
            y="Prediction",
            size="population",
            opacity=0.1,
            log_y=True,
            **facet_kwargs,
        )
    )
    fig.update_xaxes(showticklabels=True)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1 + 0.15 / len(target_selection),
            xanchor="left",
            x=0,
        )
    )
    fig.update_traces(
        marker_line=dict(width=0),
    )
    return fig


def plot_residuals(
    residuals_long, densite_population, model_selection, target_selection
):
    facet_kwargs = compute_facet_params(model_selection, target_selection)
    fig = (
        (residuals_long)
        .loc[lambda df: df.model.isin(model_selection)]
        .loc[lambda df: df.target.isin(target_selection)]
        .merge(densite_population, on="code_census_tract")
        .rename(
            columns={
                "densite_lognorm": "Density (log normalized)",
                "value_ratio": "Prediction / Truth",
            }
        )
        .plot(
            kind="scatter",
            x="Density (log normalized)",
            y="Prediction / Truth",
            size="population",
            opacity=0.1,
            log_y=True,
            **facet_kwargs,
        )
    )
    fig.update_xaxes(showticklabels=True)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1 + 0.15 / len(target_selection),
            xanchor="left",
            x=0,
        )
    )
    fig.update_traces(
        marker_line=dict(width=0),
    )
    return fig


# %%
pd.options.plotting.backend = "plotly"
startup_path = Path.cwd()
project_path = startup_path.parent
reload_kedro(project_path)
embed_kwargs = dict(include_plotlyjs="cdn", full_html=False)


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
census_metadata = catalog.load("census_metadata")

# %%
features = runs[0]["features"]
densite_population = compute_densite_population_from_features(features)

# %%
targets_and_predictions = compute_targets_and_predictions(targets, runs)
targets_and_predictions_long = compute_targets_and_predictions_long(
    targets_and_predictions
).assign(model=lambda df: (df.model).str.replace(r"seed[0-9]*_id", ""))
targets_and_predictions_per_target = compute_targets_and_predictions_per_target(
    targets_and_predictions_long
)
residuals_per_target = compute_residuals_per_target(
    targets_and_predictions_per_target,
)
residuals_long = compute_residuals_long(residuals_per_target)

# %%
target_selection = ["inscrits", "voix", "gauche", "droite", "extreme_droite"]
model_selection = [
    "model_zero_001",
    "model_minimal_003",
    "model_light_005",
    "model_complex_007",
    "model_full_009",
    "truth",
]

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    ["truth"],
    target_selection,
)
fig.write_html("truth_all_targets.embed.html", **embed_kwargs)

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    ["truth"],
    target_selection[:2],
)
fig.write_html("truth_voix_inscrits.embed.html", **embed_kwargs)

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    ["truth"],
    target_selection[2:],
)
fig.write_html("truth_gauche_droite_extreme_droite.embed.html", **embed_kwargs)

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    model_selection,
    target_selection,
)
fig.write_html("predictions_all_targets.embed.html", **embed_kwargs)

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    model_selection,
    target_selection[:2],
)
fig.write_html("predictions_voix_inscrits.embed.html", **embed_kwargs)

fig = plot_predictions(
    targets_and_predictions_long,
    densite_population,
    model_selection,
    target_selection[2:],
)
fig.write_html(
    "predictions_gauche_droite_extreme_droite.embed.html", **embed_kwargs
)

fig = plot_residuals(
    residuals_long, densite_population, model_selection, target_selection
)
fig.write_html("residuals_all_predictions.embed.html", **embed_kwargs)

fig = plot_residuals(
    residuals_long, densite_population, model_selection, target_selection[:2]
)
fig.write_html("residuals_voix_inscrits.embed.html", **embed_kwargs)

fig = plot_residuals(
    residuals_long, densite_population, model_selection, target_selection[2:]
)
fig.write_html(
    "residuals_gauche_droite_extreme_droite.embed.html", **embed_kwargs
)

# %%
model_selection = [
    "target",
    "model_full_009",
    "model_full_008",
    "model_complex_007",
    "model_complex_006",
    "model_light_005",
    "model_light_004",
    "model_minimal_003",
    "model_minimal_002",
    "model_zero_001",
    "model_zero_000",
]
fig = (
    (targets_and_predictions_per_target)
    .drop(["code_census_tract", "truth"], axis=1)
    .loc[:, model_selection]
    .corr(method="pearson")
    .pipe(px.imshow, width=800, height=600, template="plotly_white")
)
fig.write_html("correlations_all_models.embed.html", **embed_kwargs)

# %%
