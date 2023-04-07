import logging
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from company_df import get_persisted_company_df
from parquet_cache_util import persist_df, load_df

logger = logging.getLogger('risk_factor_analysis')


def create_company_weight_df():
    """
    Creates a vectorizer and an LDA model, and extracts topics and weights out of the company risk factor dataframe.
    The vectorizer and LDA model get persisted to disk since they are required later in the process.

    Returns:
    DataFrame: Pandas DataFrame with indexes: 'ticker', 'date' and a column 'risk_factor_weight_vector'
    """

    def save_pickle(obj, name):
        with open(os.path.join(model_dir, name + '.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    logger.info('Creating company risk factor weight dataframe...')
    model_dir = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('MODEL_DIR'))
    vectorizer_name = os.getenv('VECTORIZER_NAME')
    lda_model_name = os.getenv('LDA_MODEL_NAME')
    risk_factor_list_name = os.getenv('RISK_FACTOR_LIST_NAME')
    risk_factor_keyword_list_name = os.getenv('RISK_FACTOR_KEYWORD_LIST_NAME')
    n = int(os.getenv('RISK_FACTOR_TOPIC_NR'))
    n_top_words = int(os.getenv('TOP_WORD_NR', '20'))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=0.01, max_df=0.1)
    lda = LatentDirichletAllocation(n_components=n, max_iter=400, learning_method='batch', random_state=42)
    risk_factor_df = get_persisted_company_df().reset_index(drop=False)
    X = vectorizer.fit_transform(risk_factor_df['content'])
    weights = lda.fit_transform(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_top_words]
        topic_keywords.append(list(feature_names.take(top_keyword_locs)))

    risk_factor_names = ['risk_factor' + str(i) for i in range(lda.n_components)]

    save_pickle(vectorizer, vectorizer_name)
    save_pickle(lda, lda_model_name)
    save_pickle(risk_factor_names, risk_factor_list_name)
    save_pickle(topic_keywords, risk_factor_keyword_list_name)

    weight_df = pd.DataFrame(weights, columns=risk_factor_names)
    weight_df['risk_factor_weight_vector'] = weight_df.values.tolist()
    weight_df.drop(risk_factor_names, axis=1, inplace=True)

    company_weight_df = pd.concat([risk_factor_df[['ticker', 'date']], weight_df], axis=1)
    company_weight_df.reset_index(drop=True)
    company_weight_df.set_index(['ticker', 'date'], inplace=True)
    return company_weight_df


def persist_company_weight_df(df):
    """
    Persists the given dataframe to disk in a compressed Parquet format.

    Parameters:
    df (DataFrame): the company risk factor weight dataframe
    """
    logger.info('Persisting company risk factor weight dataframe...')
    persist_df(df, os.getenv('COMPANY_WEIGHT_CACHE_FILE_NAME'))


def get_persisted_company_weight_df():
    """
    Reads company risk factor weight dataframe from cache.

    Returns:
    DataFrame: Pandas DataFrame with indexes: 'ticker', 'date' and a column 'risk_factor_weight_vector'
    """
    logger.info('Loading persisted company risk factor weight dataframe...')
    return load_df(os.getenv('COMPANY_WEIGHT_CACHE_FILE_NAME'))


def get_risk_factor_names():
    """
    Reads risk factor names from a Pickle file.

    Returns:
    list: risk factor names in a list
    """
    logger.info('Loading risk factor names...')
    model_dir = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('MODEL_DIR'))
    risk_factor_list_name = os.getenv('RISK_FACTOR_LIST_NAME')
    with open(os.path.join(model_dir, risk_factor_list_name + '.pkl'), 'rb') as f:
        risk_factor_names = pickle.load(f)

    return risk_factor_names


def get_risk_factor_keyword_list():
    """
    Reads risk factor keyword lists from a Pickle file.

    Returns:
    list: list of lists of topic keywords
    """
    logger.info('Loading risk factor topic keywords...')
    model_dir = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('MODEL_DIR'))
    risk_factor_keyword_list_name = os.getenv('RISK_FACTOR_KEYWORD_LIST_NAME')
    with open(os.path.join(model_dir, risk_factor_keyword_list_name + '.pkl'), 'rb') as f:
        risk_factor_keyword_list = pickle.load(f)

    return risk_factor_keyword_list


def create_news_weight_df(news_df):
    """
    Vectorizes the given news dataframe and extracts the risk factor topic weights by using the same vectorizer
    and LDA model as for the company risk factors

    Returns:
    DataFrame: Pandas DataFrame with columns 'time', 'content', 'sentiment' and 'risk_factor_weight_vector'
    """
    logger.info('Creating news risk factor weight dataframe...')
    model_dir = os.path.join(os.getenv('DATA_BASE_DIR'), os.getenv('MODEL_DIR'))
    vectorizer_name = os.getenv('VECTORIZER_NAME')
    lda_model_name = os.getenv('LDA_MODEL_NAME')

    def load_pickle(name):
        with open(os.path.join(model_dir, name + '.pkl'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    risk_factor_names = get_risk_factor_names()
    vectorizer = load_pickle(vectorizer_name)
    lda = load_pickle(lda_model_name)
    X = vectorizer.transform(news_df['content'])
    weights = lda.transform(X)
    news_weight_df = pd.DataFrame(weights, columns=risk_factor_names)
    news_weight_df['risk_factor_weight_vector'] = news_weight_df.values.tolist()
    news_weight_df.drop(risk_factor_names, axis=1, inplace=True)
    return pd.concat([news_df, news_weight_df], axis=1)


def persist_news_weight_df(df):
    """
    Persists the given dataframe to disk in a compressed Parquet format.

    Parameters:
    df (DataFrame): the news risk factor weight dataframe
    """
    logger.info('Persisting news risk factor weight dataframe...')
    persist_df(df, os.getenv('NEWS_WEIGHT_CACHE_FILE_NAME'))


def get_persisted_news_weight_df():
    """
    Reads news risk factor weight dataframe from cache.

    Returns:
    DataFrame: Pandas DataFrame with columns 'time' and 'risk_factor_weight_vector'
    """
    logger.info('Loading persisted news risk factor weight dataframe...')
    return load_df(os.getenv('NEWS_WEIGHT_CACHE_FILE_NAME'))


def calculate_hourly_weight_similarity(company_weight_df, news_weight_df):
    """
    Merges company and news risk factor weight dataframes, calculates similariry measures between their weight vectors,
    resamples the values on an hourly basis, moves columns corresponding to hours of the day to columns,
    and returns the resulting dataframe.

    Returns:
    DataFrame: Pandas DataFrame with indexes 'ticker' and 'date' and several numeric columns
    """

    logger.info('Calculating hourly similarity between company and news weights...')
    n = int(os.getenv('RISK_FACTOR_TOPIC_NR', '10'))
    chunk_size = int(os.getenv('DF_CHUNK_SIZE', '10000'))

    def ffill_company_weight_df(company_weight_df):
        min_date = company_weight_df['date'].min()
        max_date = company_weight_df['date'].max()
        all_dates = pd.date_range(min_date, max_date, freq='D')
        unique_tickers = company_weight_df['ticker'].unique()
        ticker_date = pd.MultiIndex.from_product([unique_tickers, all_dates], names=['ticker', 'date']).to_frame(
            index=False)
        merged_df = pd.merge(ticker_date, company_weight_df, on=['ticker', 'date'], how='left')
        filled_df = merged_df.groupby('ticker', group_keys=True).apply(lambda group: group.fillna(method='ffill'))
        result_df = filled_df.reset_index(drop=True)
        return result_df.dropna()

    max_distance = math.sqrt(n)
    company_weight_df.reset_index(drop=False, inplace=True)
    company_weight_filled_df = ffill_company_weight_df(company_weight_df)
    logger.debug('Forward filling finished.')
    company_weight_filled_df['date'] = pd.to_datetime(company_weight_filled_df['date'])
    news_weight_df['date'] = pd.to_datetime(pd.to_datetime(news_weight_df['time']).dt.date)

    n_rows = len(company_weight_filled_df)
    dfs = []
    for i in range(0, n_rows, chunk_size):
        news_chunk = news_weight_df.iloc[i:i + chunk_size]
        if len(news_chunk) == 0:
            break

        logger.debug('Processing chunk ' + str(i + chunk_size))

        merged_df = pd.merge(news_chunk, company_weight_filled_df, on='date', how='left', suffixes=('_x', '_y'))
        logger.info('Merge done.')

        logger.debug('Calculating factor weights...')
        vector_x = np.array(merged_df['risk_factor_weight_vector_x'].tolist())
        vector_y = np.array(merged_df['risk_factor_weight_vector_y'].tolist())

        merged_df['euc_w'] = max_distance - np.linalg.norm(vector_x - vector_y, axis=1)
        merged_df['euc_w_sen'] = merged_df['euc_w'] * merged_df['sentiment']

        logger.debug('Factor weights calculated.')

        merged_df = merged_df.groupby(['ticker', pd.Grouper(freq='H', key='time')]).mean()
        logger.debug('Hourly grouping done')

        merged_df.reset_index(inplace=True)
        merged_df.set_index(['ticker', 'time'], inplace=True)
        merged_df['hour'] = merged_df.index.get_level_values('time').hour
        merged_df.rename(columns={'sentiment': 'sen'}, inplace=True)
        merged_df = merged_df.pivot_table(index=['ticker', 'time'],
                                          columns='hour',
                                          values=['sen', 'euc_w', 'euc_w_sen'])

        logger.debug('Pivot done.')
        merged_df.columns = [f'{col[0]}_{col[1]}' for col in merged_df.columns.values]
        merged_df.reset_index(inplace=True)
        merged_df.set_index('time', inplace=True)
        merged_df = merged_df.groupby('ticker').resample('D').sum()
        merged_df.index.names = ['ticker', 'date']
        dfs.append(merged_df)

    return pd.concat(dfs)


def persist_hourly_weight_similarity_df(df):
    """
    Persists the given dataframe to disk in a compressed Parquet format.

    Parameters:
    df (DataFrame): the hourly risk factor weight similarity dataframe
    """
    logger.info('Persisting hourly risk factor weight similarity dataframe...')
    persist_df(df, os.getenv('HOURLY_WEIGHT_SIMILARITY_CACHE_FILE_NAME'))


def get_persisted_hourly_weight_similarity_df():
    """
    Reads hourly risk factor weight similarity dataframe from cache.

    Returns:
    DataFrame: Pandas DataFrame with indexes 'ticker' and 'date' and several numeric columns
    """
    logger.info('Loading persisted hourly risk factor weight similarity dataframe...')
    return load_df(os.getenv('HOURLY_WEIGHT_SIMILARITY_CACHE_FILE_NAME'))
