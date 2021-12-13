from slugify import slugify


def compute_census_metadata(df_census_metadata_raw):
    rename_dict = {
        "COD_VAR": "variable_code",
        "LIB_VAR": "variable_description",
        "COD_MOD": "value_code",
        "LIB_MOD": "value_description",
        "TYPE_VAR": "variable_type",
        "LONG_VAR": "variable_length",
    }
    stopwords = [
        "un",
        "d",
        "de",
        "la",
        "le",
        "ou",
        "l",
        "dans",
        "en",
        "au",
        "une",
        "et",
        "du",
        "a",
        "quant",
    ]
    df_census_metadata = (
        (df_census_metadata_raw)
        .rename(columns=rename_dict)
        .assign(
            variable_name=lambda df: (df.variable_description)
            .str.replace(r"[\-\.]", "")
            .apply(slugify, stopwords=stopwords, separator="_")
            .str.replace(r"triris", "TRIRIS")
            .str.replace(r"iris", "IRIS")
            .str.replace(r"domtomcom", "DOM_TOM_COM")
            .str.replace(r"dom", "DOM")
            .str.replace(r"cantonouville", "canton_ou_ville")
            .str.replace(r"hlm", "HLM")
        )
    )
    return df_census_metadata
