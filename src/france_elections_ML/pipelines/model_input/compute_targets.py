# %%
import pandas as pd
import numpy as np


def compute_votes_nuance_by_bureau(df_municipales_2020_t1):
    return (
        (df_municipales_2020_t1)
        .groupby(
            ["code_commune", "code_bureau_vote", "code_nuance"], as_index=False
        )
        .agg({"inscrits": "max", "exprimes": "max", "voix": "sum"})
        .assign(
            voix=lambda df: np.where(
                df.code_nuance == "NC", df.exprimes, df.voix
            )
        )
        .drop("exprimes", axis=1)
    )


def compute_votes_gauche_droite_by_bureau(df):
    df_gauche_droite = pd.DataFrame(
        [
            ("LEXG", "Extrême gauche", "gauche"),
            ("LCOM", "Parti communiste français", "gauche"),
            ("LFI", "La France insoumise", "gauche"),
            ("LSOC", "Parti socialiste", "gauche"),
            ("LRDG", "Parti radical de gauche", "gauche"),
            ("LDVG", "Divers gauche", "gauche"),
            ("LUG", "Union de la gauche", "gauche"),
            ("LVEC", "Europe Ecologie-Les Verts", "gauche"),
            ("LECO", "Ecologiste", "gauche"),
            ("LDIV", "Divers", "autre"),
            ("LREG", "Régionaliste", "autre"),
            ("LGJ", "Gilets jaunes", "autre"),
            ("LREM", "La République en marche", "droite"),
            ("LMDM", "Modem", "droite"),
            ("LUDI", "Union des Démocrates et Indépendants", "droite"),
            ("LUC", "Union du centre", "droite"),
            ("LDVC", "Divers centre", "droite"),
            ("LLR", "Les Républicains", "droite"),
            ("LUD", "Union de la droite", "droite"),
            ("LDVD", "Divers droite", "droite"),
            ("LDLF", "Debout la France", "droite"),
            ("LRN", "Rassemblement National", "droite"),
            ("LEXD", "Extrême droite", "droite"),
            ("LNC", "Non Communiqué", "autre"),
            ("NC", "Non Communiqué", "autre"),
        ],
        columns=["code_nuance", "nom_parti", "gauche_droite"],
    )
    return (
        (df)
        .merge(
            df_gauche_droite.loc[:, ["code_nuance", "gauche_droite"]],
            on="code_nuance",
        )
        .groupby(
            ["code_commune", "code_bureau_vote", "gauche_droite"],
            as_index=False,
        )
        .agg({"voix": "sum", "inscrits": "max"})
    )


def aggregate_votes_gauche_droite_by_census_tract(df, df_census_tract_code):
    return (
        (df)
        .merge(
            df_census_tract_code.loc[:, ["code_commune", "code_census_tract"]],
            on="code_commune",
            how="inner",
        )
        .pivot_table(
            index=["code_census_tract", "code_bureau_vote", "inscrits"],
            columns="gauche_droite",
            values="voix",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
        .groupby(["code_census_tract"], as_index=False)
        .agg(
            {
                "autre": "sum",
                "droite": "sum",
                "gauche": "sum",
                "inscrits": "sum",
            }
        )
        .assign(voix=lambda df: df.autre + df.gauche + df.droite)
        .sort_values(by="code_census_tract")
        .reset_index(drop=True)
    )


def compute_targets(df_municipales_2020_t1, df_census_tract_code):
    df_targets = (
        (df_municipales_2020_t1)
        .pipe(compute_votes_nuance_by_bureau)
        .pipe(compute_votes_gauche_droite_by_bureau)
        .pipe(
            aggregate_votes_gauche_droite_by_census_tract, df_census_tract_code
        )
    )
    return df_targets
