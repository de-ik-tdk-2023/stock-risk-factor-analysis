import glob
import logging
import os

import pandas as pd

logger = logging.getLogger('risk_factor_analysis')


def create_news_df():
    """
    Reads preprocessed news from Parquet files and creates a Pandas DataFrame out of them.

    Returns:
    DataFrame: Pandas DataFrame with columns 'time', 'content' and sentiment
    """
    logger.info('Creating news dataframe...')
    path = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('PREPROCESSED_NEWS_DIR'), '*.parquet.gzip')
    news_files = glob.glob(path)
    dfs = []
    for news_file in news_files:
        df = pd.read_parquet(news_file)
        df['time'] = df['date'] + pd.Timedelta(hours=8)
        dfs.append(df)

    df = pd.concat(dfs)[['time', 'content', 'sentiment']]
    df.reset_index(drop=True, inplace=True)
    return df
