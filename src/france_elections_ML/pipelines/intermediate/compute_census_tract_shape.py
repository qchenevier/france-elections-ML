import numpy as np


def compute_commune_shape(IRIS_shape_raw):
    IRIS_shape = (
        (IRIS_shape_raw)
        .rename(
            columns=lambda c: (c)
            .lower()
            .replace("typ", "type")
            .replace("iris", "IRIS")
            .replace("com", "commune")
            .replace("insee", "code")
            .replace("nom", "libelle")
        )
        .assign(
            libelle=lambda df: df.libelle_commune
            + np.where(
                df.libelle_commune != df.libelle_IRIS,
                df.libelle_IRIS.apply(" ({})".format),
                "",
            )
        )
    )
    commune_shape = (
        (IRIS_shape)
        .loc[:, ["code_commune", "libelle_commune", "geometry"]]
        .dissolve(by="code_commune", as_index=False)
    )
    return commune_shape


def clean_text_for_plotly(serie):
    return serie.str.wrap(50).str.replace("\n", "<br>")


def compute_census_tract_shape(commune_shape, df_census_tract_code):
    census_tract_shape = (
        (commune_shape)
        .merge(df_census_tract_code, on="code_commune", how="inner")
        .groupby("code_census_tract", as_index=False)
        .apply(lambda df: df.assign(libelle=", ".join(df.libelle_commune)))
        .assign(libelle=lambda df: clean_text_for_plotly(df.libelle))
        .loc[
            :,
            [
                "code_census_tract",
                "libelle",
                "geometry",
            ],
        ]
        .dissolve(by="code_census_tract", as_index=False)
        .assign(
            surface=lambda df: (df.geometry)
            .set_crs(epsg=3395)
            .map(lambda p: p.area / 10 ** 6)
        )
    )
    return census_tract_shape
