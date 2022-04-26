# %%
import pandas as pd
import numpy as np


def compute_votes_by_candidate_by_bureau(df_presidentielles_2022_t1):
    return (
        (df_presidentielles_2022_t1)
        .groupby(
            ["code_commune", "code_bureau_vote", "nom", "prenom"],
            as_index=False,
        )
        .agg({"inscrits": "max", "voix": "sum"})
    )


def compute_votes_gauche_droite_by_candidate_by_bureau(df):
    df_gauche_droite = pd.DataFrame(
        [
            ("ARTHAUD", "Nathalie", "gauche"),
            ("ROUSSEL", "Fabien", "gauche"),
            ("MACRON", "Emmanuel", "droite"),
            ("LASSALLE", "Jean", "droite"),
            ("LE PEN", "Marine", "extreme_droite"),
            ("ZEMMOUR", "Éric", "extreme_droite"),
            ("MÉLENCHON", "Jean-Luc", "gauche"),
            ("HIDALGO", "Anne", "gauche"),
            ("JADOT", "Yannick", "gauche"),
            ("PÉCRESSE", "Valérie", "droite"),
            ("POUTOU", "Philippe", "gauche"),
            ("DUPONT-AIGNAN", "Nicolas", "extreme_droite"),
        ],
        columns=["nom", "prenom", "gauche_droite"],
    )
    return (
        (df)
        .merge(
            df_gauche_droite.loc[:, ["nom", "prenom", "gauche_droite"]],
            on=["nom", "prenom"],
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
                "extreme_droite": "sum",
                "droite": "sum",
                "gauche": "sum",
                "inscrits": "sum",
            }
        )
        .assign(voix=lambda df: df.extreme_droite + df.gauche + df.droite)
        .sort_values(by="code_census_tract")
        .reset_index(drop=True)
    )


def compute_targets(df_presidentielles_2022_t1, df_census_tract_code):
    df_targets = (
        (df_presidentielles_2022_t1)
        .pipe(compute_votes_by_candidate_by_bureau)
        .pipe(compute_votes_gauche_droite_by_candidate_by_bureau)
        .pipe(
            aggregate_votes_gauche_droite_by_census_tract, df_census_tract_code
        )
    )
    return df_targets
