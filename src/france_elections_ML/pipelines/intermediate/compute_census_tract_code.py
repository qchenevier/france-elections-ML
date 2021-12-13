# %%
import polars as pl
import numpy as np


def compute_commune_code_in_census(df_census):
    census_tract_columns = [
        "departement_canton_ou_ville_lieu_residence_pseudocanton",
        "code_IRIS_lieu_residence",
    ]
    df_canton_pseudocommune_code = (
        (df_census)
        .select(census_tract_columns)
        .drop_duplicates()
        .with_column(pl.col(census_tract_columns[0]).alias("code_canton"))
        .with_column(
            pl.col(census_tract_columns[1])
            .str.slice(0, 5)
            .alias("code_pseudocommune")
        )
        .with_column(
            (
                pl.col("code_canton")
                + pl.lit("_")
                + pl.col("code_pseudocommune")
            ).alias("code_canton_pseudocommune")
        )
        .drop(census_tract_columns)
        .drop_duplicates()
        .sort(["code_canton_pseudocommune"])  # needed for reproducibility
        .to_pandas()
    )
    df_commune_code_in_census = (
        (df_canton_pseudocommune_code)
        .rename(columns={"code_pseudocommune": "code_commune"})
        .loc[lambda df: df.code_commune != "ZZZZZ"]
        .loc[:, ["code_canton", "code_commune"]]
    )
    return df_commune_code_in_census


def compute_commune_code(df_commune_code_raw, df_commune_code_in_census):
    return (
        (df_commune_code_raw)
        .loc[lambda df: df.typecom == "COM"]
        .rename(columns={"com": "code_commune", "can": "code_canton"})
        .pipe(
            lambda df: df.fillna({"code_canton": df.dep.apply("{}ZZ".format)})
        )
        .loc[:, ["code_canton", "code_commune"]]
        .append(df_commune_code_in_census)
        .drop_duplicates()
    )


def compute_census_tract_code(df_commune_code, df_commune_code_in_census):
    census_commune_codes = df_commune_code_in_census.code_commune.unique()
    df_census_tract_code = (
        (df_commune_code)
        .assign(
            code_census_commune=lambda df: np.where(
                df.code_commune.isin(census_commune_codes),
                df.code_commune,
                "ZZZZZ",
            )
        )
        .assign(
            code_census_tract=lambda df: df.code_canton
            + "_"
            + df.code_census_commune
        )
    )
    return df_census_tract_code
