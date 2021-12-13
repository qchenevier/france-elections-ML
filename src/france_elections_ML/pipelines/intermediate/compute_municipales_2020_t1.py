import io
import re

import numpy as np
import pandas as pd
from slugify import slugify


def fix_text_in_virtual_file(f, pattern, replacement):
    virtual_file = io.StringIO()
    for line in f.readlines():
        if re.search(pattern, line):
            print("pattern found")
            virtual_file.write(re.sub(pattern, replacement, line))
        else:
            virtual_file.write(line)
    virtual_file.seek(0)
    f.close()
    return virtual_file


def clean_ratio_column(df):
    df = df.copy()
    for c in df.filter(regex="ratio").columns:
        df[c] = df[c].str.replace(",", ".").astype("float64") / 100
    return df


def clean_int_columns(df, int_columns):
    df = df.copy()
    for c in int_columns:
        df[c] = df[c].astype(int)
    return df


def fix_paris_arrondissements(df):
    return df.assign(
        libelle_commune=lambda df: np.where(
            df.code_departement == 75,
            df.code_bureau_vote.astype(str)
            .str[:-2]
            .apply("Paris {}e Arrondissement".format),
            df.libelle_commune,
        )
    ).assign(
        code_commune=lambda df: np.where(
            df.code_departement == 75,
            df.code_bureau_vote.astype(str).str[:-2].apply("1{:0>2}".format),
            df.code_commune,
        )
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
            df.code_bureau_vote.astype(str).str[:-2].apply("38{:0>1}".format),
            df.code_commune,
        )
    )


def fix_marseille_arrondissements(df):
    return df.assign(
        libelle_commune=lambda df: np.where(
            (df.code_departement == 13) & (df.code_commune == 55),
            df.code_bureau_vote.astype(str)
            .str.replace(r"\.0$", "")
            .str[:-2]
            .apply("Marseille {}e Arrondissement".format),
            df.libelle_commune,
        )
    ).assign(
        code_commune=lambda df: np.where(
            (df.code_departement == 13) & (df.code_commune == 55),
            df.code_bureau_vote.astype(str)
            .str.replace(r"\.0$", "")
            .str[:-2]
            .apply("2{:0>2}".format),
            df.code_commune,
        )
    )


def fix_code_bureau_vote(df):
    df = df.copy()
    df.code_bureau_vote = np.where(
        df.code_bureau_vote == "0,00E+00",
        df.code_bureau_vote.astype(str)
        + (df.code_bureau_vote == "0,00E+00").cumsum().astype(str),
        df.code_bureau_vote,
    )
    return df


def wide_to_long(df, candidate_columns, station_columns):
    return (
        df.pipe(
            pd.wide_to_long,
            stubnames=candidate_columns,
            i=station_columns,
            j="no_candidat",
            sep="_",
        )
        .dropna(subset=["no_panneau"])
        .reset_index()
    )


def make_steps(df, step_size=1000):
    return (
        pd.Series(list(range(df.shape[0])))
        .pipe(lambda s: (s - (s % step_size)) / step_size)
        .astype(int)
    )


def compute_municipales_2020_t1_raw_fixed(text):
    return (
        (text)
        .replace(
            "LE LARDIN ST LAZARE \t un nouveau",
            "LE LARDIN ST LAZARE un nouveau",
        )
        .replace(
            "ENSEMBLE POUR BRASPARTS \t BRASPARZH D'AN HOLL",
            "ENSEMBLE POUR BRASPARTS BRASPARZH D'AN HOLL",
        )
    )


# %%
def compute_municipales_2020_t1(df_municipales_2020_t1_raw):
    columns_raw = df_municipales_2020_t1_raw.columns.tolist()

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
    station_columns = columns[:19]
    candidate_columns = columns[19:28]

    # %%
    df_municipales_2020_t1_wide = (
        (df_municipales_2020_t1_raw)
        .set_axis(columns, axis=1)
        .pipe(fix_paris_arrondissements)
        .pipe(fix_lyon_arrondissements)
        .pipe(fix_marseille_arrondissements)
    )

    # %%
    df_municipales_2020_t1 = (
        (df_municipales_2020_t1_wide)
        .rename(columns={c: f"{c}_0" for c in candidate_columns})
        .pipe(fix_code_bureau_vote)
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

    return df_municipales_2020_t1
