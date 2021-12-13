# %%
import geopandas as gpd

from .utils import download_and_extract_dataset


# %%
def download_and_extract_census_raw():
    # official page: https://www.insee.fr/fr/statistiques/4802064
    return download_and_extract_dataset(
        url="https://www.insee.fr/fr/statistiques/fichier/4802064/RP2017_INDCVI_csv.zip",
        extract=["FD_INDCVI_2017.csv", "varmod_INDCVI_2017.csv"],
    )


def download_and_extract_commune_code_raw():
    # official page: https://www.insee.fr/fr/information/4316069
    return download_and_extract_dataset(
        url="https://www.insee.fr/fr/statistiques/fichier/4316069/communes2020-csv.zip",
        extract="communes2020.csv",
    )


def download_and_extract_IRIS_shape_raw():
    return download_and_extract_dataset(
        url="ftp://Contours_IRIS_ext:ao6Phu5ohJ4jaeji@ftp3.ign.fr/CONTOURS-IRIS_2-1__SHP__FRA_2020-01-01.7z",
        extract=(
            "CONTOURS-IRIS_2-1__SHP__FRA_2020-01-01/CONTOURS-IRIS/"
            "1_DONNEES_LIVRAISON_2020-12-00282/CONTOURS-IRIS_2-1_SHP_LAMB93_FXX-2020/"
            "CONTOURS-IRIS.shp"
        ),
        read_func=gpd.read_file,
    )


def download_and_extract_municipales_2020_t1():
    # official page: https://www.data.gouv.fr/en/datasets/elections-municipales-2020-resultats-1er-tour/
    return download_and_extract_dataset(
        url=(
            "https://static.data.gouv.fr/resources/"
            "elections-municipales-2020-resultats/20200525-133745/"
            "2020-05-18-resultats-par-niveau-burvot-t1-france-entiere.txt"
        ),
        read_kwargs=dict(encoding="ISO-8859-1"),
    )


# %%
