import logging


def compute_census(df_census_raw, df_census_metadata):
    log = logging.getLogger(__name__)
    rename_dict = (
        (df_census_metadata)
        .loc[:, ["variable_code", "variable_name"]]
        .drop_duplicates()
        .set_index("variable_code")
        .variable_name.to_dict()
    )

    df = df_census_raw.rename(rename_dict)

    log.info("Census: sorting dataframe")
    df.sort(df.columns, in_place=True)
    return df
