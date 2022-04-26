# %%
from pathlib import Path
import re
import logging
import warnings


import pandas as pd
import polars as pl
import torch
import shap
import matplotlib as mpl
import xarray

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


def functionalize_model(model, rounded=True, target_idx=None):
    def model_wrapped(X):
        prediction = (
            model.model(torch.tensor(X))
            .detach()
            .numpy()
            .reshape(-1, 5)[
                :, target_idx if target_idx is not None else slice(None)
            ]
        )
        if rounded:
            return prediction.round()
        return prediction

    return model_wrapped


def pad_suffix(series, width=2):
    prefix = (series).str.split("_").str[:-1].str.join("_")
    suffix_padded = (
        (series).str.split("_").str[-1].str.pad(width=width, fillchar="0")
    )
    return prefix + "_" + suffix_padded


def pad_variable_code_in_feature_name(df, variable_lengths):
    return (
        df.assign(
            variable_name=lambda df: df.feature_name.str.split("_")
            .str[:-1]
            .str.join("_")
        )
        .assign(variable_code=lambda df: df.feature_name.str.split("_").str[-1])
        .groupby("variable_name", as_index=False)
        .apply(
            lambda df: df.assign(
                variable_code_padded=df.variable_code.str.pad(
                    width=variable_lengths.get(df.variable_name.iloc[0], 0),
                    fillchar="0",
                )
            )
        )
        .reset_index(drop=True)
        .assign(
            feature_name=lambda df: df.variable_name
            + "_"
            + df.variable_code_padded
        )
        .drop(
            ["variable_name", "variable_code", "variable_code_padded"], axis=1
        )
    )


def compute_shap_values_and_features_for_target(
    shap_values, X_to_explain, feature_descriptions, targets, variable_lengths
):
    X_to_explain_long = (
        xarray.DataArray(
            X_to_explain,
            dims=["explanation_idx", "feature_name"],
            name="feature_value",
        )
        .to_dataframe()
        .reset_index()
    )
    shap_values_long = (
        xarray.DataArray(
            shap_values,
            dims=["target", "explanation_idx", "feature_name"],
            name="shap_value",
        )
        .to_dataframe()
        .reset_index()
        .assign(
            feature_name=lambda df: df.feature_name.replace(
                pd.Series(X_to_explain.columns).to_dict()
            )
        )
        .assign(
            target=lambda df: df.target.replace(
                pd.Series(targets.columns[1:]).to_dict()
            )
        )
    )
    shap_values_and_features = (
        (shap_values_long)
        .merge(X_to_explain_long, on=["explanation_idx", "feature_name"])
        .pipe(pad_variable_code_in_feature_name, variable_lengths)
        .merge(feature_descriptions, on="feature_name")
    )
    return shap_values_and_features


def compute_features_colors(series):
    features_values = series.sort_values().drop_duplicates()
    features_values_normalized = features_values.pipe(lambda s: s / s.max())
    features_colors = {
        n: mpl.colors.rgb2hex(c)
        for n, c in zip(features_values, mpl.cm.bwr(features_values_normalized))
    }
    return features_colors


def compute_shap_values_and_features(
    run,
    targets,
    feature_descriptions,
    n_background_data_samples=100,
    n_explanations=200,
):
    features = run["features"]
    model = run["model"]

    X = features.to_pandas().iloc[:, 2:]
    X_background_data = (
        (X).sample(n_background_data_samples).reset_index(drop=True)
    )
    X_to_explain = X.sample(n_explanations).reset_index(drop=True)

    f_model = functionalize_model(model, rounded=False)
    explainer = shap.KernelExplainer(
        model=f_model, data=X_background_data, algorithm="deep"
    )
    shap_values = explainer.shap_values(X=X_to_explain, nsamples=100)

    shap_values_and_features = compute_shap_values_and_features_for_target(
        shap_values,
        X_to_explain,
        feature_descriptions,
        targets,
        variable_lengths,
    )
    return shap_values_and_features


def plot_summary(shap_values_and_features, clip=None):
    df_to_plot = (
        (shap_values_and_features)
        .groupby("feature_name", as_index=False)
        .apply(
            lambda dfg: dfg.assign(
                feature_value_normalized=lambda df: df.feature_value.pipe(
                    lambda s: (s - s.min()) / (max(1, s.max()) - s.min())
                )
            )
        )
        .reset_index(drop=True)
        .assign(
            shap_value=lambda df: df.shap_value.clip(lower=-clip, upper=clip)
        )
    )
    features_colors = compute_features_colors(
        df_to_plot.feature_value_normalized
    )
    features_descriptions = (
        (df_to_plot)
        .sort_values(by="feature_name")
        .feature_description.drop_duplicates()
        .tolist()
    )
    target_order = [
        "inscrits",
        "voix",
        "gauche",
        "droite",
        "extreme_droite",
    ]
    strip_size = 20
    fig = px.strip(
        df_to_plot,
        y="feature_description",
        x="shap_value",
        color="feature_value_normalized",
        stripmode="overlay",
        width=1650,
        category_orders={
            "feature_description": features_descriptions,
            "target": target_order,
        },
        color_discrete_map=features_colors,
        template="plotly_white",
        height=strip_size * len(features_descriptions) + 140,
        facet_col="target",
    )
    (fig).update_layout(showlegend=False).update_traces(
        width=1.7, marker=dict(size=4)
    )
    return fig


def insert_row(df, row_tuple):
    return pd.concat(
        [df, pd.DataFrame([row_tuple], columns=df.columns)]
    ).reset_index(drop=True)


# %%
logging.getLogger("shap").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
pd.options.plotting.backend = "plotly"
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
census_metadata = catalog.load("census_metadata")

# %%
feature_descriptions = (
    (census_metadata)
    .assign(feature_name=lambda df: df.variable_name + "_" + df.value_code)
    .assign(
        feature_description=lambda df: df.value_description
        + " — "
        + df.variable_description
    )
    .filter(like="feature")
    .dropna()
    .pipe(insert_row, ("densite_lognorm", "Densité (log normalisée)"))
)
variable_lengths = (
    (census_metadata)
    .loc[:, ["variable_name", "variable_length"]]
    .drop_duplicates()
    .set_index("variable_name")
    .variable_length.astype(int)
    .to_dict()
)

# %%
model_selection = [
    "model_full_seed1000_id009",
    "model_complex_seed1000_id007",
    "model_light_seed1000_id005",
    "model_minimal_seed1000_id003",
    "model_zero_seed1000_id001",
]
runs_selection = {
    model_name: next(filter(lambda r: r["model_name"] == model_name, runs))
    for model_name in model_selection
}

# %%
shap_values_and_features_for_selection = {
    model_name: compute_shap_values_and_features(
        run, targets, feature_descriptions
    )
    for model_name, run in runs_selection.items()
}

# %%
clip = 0.35
for model_name in model_selection:
    logging.info("Saving explanation plot for model: %s", model_name)
    fig = plot_summary(
        shap_values_and_features_for_selection[model_name], clip=clip
    )
    fig.write_html(
        f"shap_explanations_{model_name}.embed.html",
        include_plotlyjs="cdn",
        full_html=False,
    )

# %%
