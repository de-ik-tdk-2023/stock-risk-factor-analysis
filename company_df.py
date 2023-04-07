import glob
import logging
import os
import re
from datetime import datetime

import pandas as pd

from parquet_cache_util import persist_df, load_df
from text_cleaning import preprocess_text

logger = logging.getLogger('risk_factor_analysis')


def create_company_df():
    """
    Reads risk factor text files from the file system and creates a Pandas DataFrame out of them.

    Returns:
    DataFrame: Pandas DataFrame with indexes: 'ticker', 'date' and a column 'content'
    """
    logger.info('Creating company risk factor dataframe...')
    path = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('RISK_FACTOR_TEXT_DIR'), '*.txt')
    risk_factor_files = glob.glob(path)

    df = pd.DataFrame([], columns=['ticker', 'date', 'content'])
    for risk_factor_file in risk_factor_files:
        filename = os.path.basename(risk_factor_file)
        logger.info('Processing file ', filename)
        search = re.search('([a-z]+)-10k-([0-9]{8})\.txt', filename, re.IGNORECASE)
        ticker = search.group(1)
        date = datetime.strptime(search.group(2), '%Y%m%d')
        with open(risk_factor_file, "rb") as f:
            file_content = f.read()
            file_content = file_content.decode("utf-8", "replace")
            file_content = preprocess_text(file_content)
        df = pd.concat([df, pd.Series({'ticker': ticker, 'date': date, 'content': file_content}).to_frame().T],
                       ignore_index=True)

    df.set_index(['ticker', 'date'], inplace=True)
    return df


def persist_company_df(df):
    """
    Persists the given dataframe to disk in a compressed Parquet format.

    Parameters:
    df (DataFrame): the company risk factor dataframe
    """
    logger.info('Persisting company risk factor dataframe...')
    persist_df(df, os.getenv('COMPANY_CACHE_FILE_NAME'))


def get_persisted_company_df():
    """
    Reads company risk factor dataframe from cache.

    Returns:
    DataFrame: Pandas DataFrame with indexes: 'ticker', 'date' and a column 'content'
    """
    logger.info('Loading persisted company risk factor dataframe...')
    return load_df(os.getenv('COMPANY_CACHE_FILE_NAME'))
