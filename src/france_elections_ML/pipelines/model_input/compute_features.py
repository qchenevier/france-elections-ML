# %%
from itertools import accumulate
from collections import OrderedDict

import numpy as np
import polars as pl


def get_dummies_on_columns(df, columns):
    return df.hstack(df.select(columns).to_dummies()).drop(columns)


def put_columns_first(df, columns_first):
    assert isinstance(columns_first, list)
    columns = df.columns
    columns_last = sorted([c for c in columns if c not in columns_first])
    return df.select(columns_first + columns_last)


def compute_features_per_census_tract(df_census, features):
    census_tract_columns = [
        "departement_canton_ou_ville_lieu_residence_pseudocanton",
        "code_IRIS_lieu_residence",
    ]
    groupby = census_tract_columns + features
    return (
        (df_census)
        .rename({"poids_individu": "population"})
        .groupby(groupby)
        .agg([pl.col("population").sum().alias("population")])
        .with_column(
            (
                pl.col(census_tract_columns[0])
                + pl.lit("_")
                + pl.col(census_tract_columns[1]).str.slice(0, 5)
            ).alias("code_census_tract")
        )
        .drop(census_tract_columns)
        .groupby(["code_census_tract"] + features)
        .agg([pl.col("population").sum().alias("population")])
        .sort(["code_census_tract"] + features)  # needed for reproducibility
        .pipe(get_dummies_on_columns, features)
        .pipe(put_columns_first, ["code_census_tract", "population"])
    )


def compute_density(df_census_features, census_tract_shape):
    df_surface = census_tract_shape.loc[:, ["code_census_tract", "surface"]]
    df_density = (
        (df_census_features)
        .groupby("code_census_tract")
        .agg([pl.col("population").sum().alias("population")])
        .join(pl.DataFrame(df_surface), on="code_census_tract", how="inner")
        .with_column(
            (pl.col("population") / pl.col("surface")).alias("densite")
        )
        .with_column((pl.col("densite").apply(np.log10)).alias("densite_log"))
        .with_column(
            (pl.col("densite_log") / pl.mean("densite_log")).alias(
                "densite_lognorm"
            )
        )
        .select(["code_census_tract", "densite_lognorm"])
    )
    return (df_census_features).join(
        df_density, on="code_census_tract", how="inner"
    )


def compute_features_list(param_features):
    """returns features list

    Args:
        param_features (String or list[String]):
        - String in ["zero", "minimal", "light", "complex", "full"]
        - list[String] of features names

    E.g:
        compute_features_list("zero")
        compute_features_list("minimal")
        compute_features_list("light")
        compute_features_list("complex")
        compute_features_list("full")
        compute_features_list(
            [
                "age_annees_revolues_age_dernier_anniversaire_13_classes_age_detaillees_autour_20_ans",
                "sexe",
                "statut_conjugal",
                "condition_emploi",
                "diplome_plus_eleve",
                "nombre_personnes_menage_regroupe",
                "nombre_voitures_menage",
                "statut_occupation_detaille_logement",
                "superficie_logement",
                "activite_economique_17_postes_na_a17",
                "categorie_socioprofessionnelle_8_postes",
                "temps_travail",
                "type_activite",
            ]
        )
    """
    if isinstance(param_features, list):
        return param_features
    features_groups = OrderedDict(
        zero=["sexe"],
        minimal=[
            "indicateur_nationalite_condense_francais_etranger",
            "statut_conjugal",
            "appartenance_logement_organisme_HLM",
        ],
        light=["age_quinquennal_annees_revolues"],
        complex=[
            "diplome_plus_eleve",
            "categorie_socioprofessionnelle_8_postes",
        ],
        full=[
            "activite_economique_17_postes_na_a17",
            "mode_transport_principal_plus_souvent_utilise_pour_aller_travailler",
        ],
    )
    features_accumulated = dict(
        zip(features_groups.keys(), accumulate(features_groups.values()))
    )
    return features_accumulated[param_features]


def compute_features(df_census, census_tract_shape, param_features):
    features = compute_features_list(param_features)
    df_features = (
        (df_census)
        .pipe(compute_features_per_census_tract, features)
        .pipe(compute_density, census_tract_shape)
    )
    return df_features
