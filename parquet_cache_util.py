import os

import pandas as pd


def persist_df(df, file_name_prefix):
    """
    Persists the given dataframe to disk in a compressed Parquet format.

    Parameters:
    df (DataFrame): the dataframe to persist
    file_name_prefix (str): the prefix of the filename without .parquet.gzip
    """
    cache_dir = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('CACHE_DIR'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    path = os.path.join(cache_dir, file_name_prefix + '.parquet.gzip')
    df.to_parquet(path, compression="gzip")


def load_df(file_name_prefix):
    """
    Reads dataframe with the given filename prefix from cache.

    Parameters:
    file_name_prefix (str): the prefix of the filename without .parquet.gzip

    Returns:
    DataFrame: Pandas DataFrame
    """
    path = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('CACHE_DIR'), file_name_prefix + '.parquet.gzip')
    return pd.read_parquet(path)
