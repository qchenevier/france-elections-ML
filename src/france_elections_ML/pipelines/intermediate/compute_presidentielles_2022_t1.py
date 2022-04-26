import numpy as np
import pandas as pd
from slugify import slugify

from .compute_municipales_2020_t1 import (
    clean_int_columns,
    clean_ratio_column,
    fix_paris_arrondissements,
    fix_marseille_arrondissements,
    wide_to_long,
    make_steps,
)


def fix_lyon_arrondissements(df):
    return df.assign(
        libelle_commune=lambda df: np.where(
            (df.code_departement == 69) & (df.code_commune == 123),
            df.code_bureau_vote.astype(str)
            .str[:-2]
            .apply("Lyon {}e Arrondissement".format),
            df.libelle_commune,
        )
    ).assign(
        code_commune=lambda df: np.where(
            (df.code_departement == 69) & (df.code_commune == 123),
            df.code_bureau_vote.astype(str).str[1:-2].apply("38{:0>1}".format),
            df.code_commune,
        )
    )


def compute_presidentielles_2022_t1_raw_fixed(text):
    text_lines = text.split("\n")
    header = text_lines[0]
    columns = header.split(";")
    station_columns = columns[:21]
    candidate_columns = columns[21:28]
    columns_fixed = station_columns + 12 * candidate_columns
    header_fixed = ";".join(columns_fixed)
    text_lines[0] = header_fixed
    return "\n".join(text_lines)


def compute_presidentielles_2022_t1(df_presidentielles_2022_t1_raw):
    columns_raw = df_presidentielles_2022_t1_raw.columns.tolist()

    # %%
    replacements = [
        ["%", "ratio"],
        ["n°", "no "],
        ["b.vote", "bureau vote"],
        ["n.pan", "no_panneau"],
    ]
    stopwords = ["de", "du", "le", "la"]

    columns = (
        pd.Series(
            [
                slugify(
                    c.lower(),
                    separator=" ",
                    replacements=replacements,
                    stopwords=stopwords,
                )
                for c in columns_raw
            ]
        )
        .str.replace(r"\bexp\b", "exprimes")
        .str.replace(r"\bins\b", "inscrits")
        .str.replace(r"\bvot\b", "votants")
        .str.replace(r"\babs\b", "abstentions")
        .str.replace(" ", "_")
        .tolist()
    )

    # %%
    station_columns = columns[:21]
    candidate_columns = columns[21:28]

    # %%
    df_presidentielles_2022_t1_wide = (
        (df_presidentielles_2022_t1_raw)
        .set_axis(columns, axis=1)
        .pipe(fix_paris_arrondissements)
        .pipe(fix_lyon_arrondissements)
        .pipe(fix_marseille_arrondissements)
    )

    # %%
    df_presidentielles_2022_t1 = (
        (df_presidentielles_2022_t1_wide)
        .rename(columns={c: f"{c}_0" for c in candidate_columns})
        .assign(step=lambda df: make_steps(df, step_size=10000))
        .groupby("step", as_index=False)
        .apply(wide_to_long, candidate_columns, station_columns)
        .drop("step", axis=1)
        .reset_index(drop=True)
        .pipe(clean_ratio_column)
        .pipe(clean_int_columns, int_columns=["no_panneau", "voix"])
        .assign(code_departement=lambda df: df.code_departement.astype(str))
        .assign(code_bureau_vote=lambda df: df.code_bureau_vote.astype(str))
        .assign(
            code_commune=lambda df: df.code_departement.apply("{:0>2}".format)
            + df.code_commune.apply("{:0>3}".format)
        )
    )

    return df_presidentielles_2022_t1
