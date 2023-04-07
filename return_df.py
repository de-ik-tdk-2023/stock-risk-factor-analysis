import glob
import logging
import os
import re

import numpy as np
import pandas as pd

import financial_periods as per

logger = logging.getLogger('risk_factor_analysis')


def create_return_df():
    """
    Reads price data for all available tickers, combines that with benchmark data and returns the DataFrame.

    Returns:
    DataFrame: Pandas DataFrame with indexes: 'ticker', 'date' and several numeric columns
    """
    logger.info('Creating return dataframe...')
    data_base_dir = os.getenv('DATA_BASE_DIR')
    index_dir = os.getenv('INDEX_DIR')
    price_dir = os.getenv('PRICE_DIR')
    sp500_df = pd.read_csv(os.path.join(data_base_dir, index_dir, 'sp500.csv'))
    sp500_df['date'] = pd.to_datetime(sp500_df['Date']).dt.tz_localize(None)
    sp500_df['bench_return'] = sp500_df['Close'].pct_change()

    dfs = []
    files = glob.glob(os.path.join(data_base_dir, price_dir, '*.csv'))
    for file in files:
        filename = os.path.basename(file)
        search = re.search('([a-z]+)\.csv', filename, re.IGNORECASE)
        ticker = search.group(1)
        df = pd.read_csv(file)
        df['ticker'] = ticker
        df['return'] = df['Close'].pct_change()
        df['volume'] = df['Volume']
        df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.merge(sp500_df, on='date')

        df['prev_day_volume'] = df['volume'].shift(1)
        df['weekly_volume'] = df['prev_day_volume'].rolling(window=per.W).mean()
        df['monthly_volume'] = df['prev_day_volume'].rolling(window=per.M).mean()
        df['annual_volume'] = df['prev_day_volume'].rolling(window=per.Y).mean()
        df['3yr_volume'] = df['prev_day_volume'].rolling(window=per.Y3).mean()
        df['5yr_volume'] = df['prev_day_volume'].rolling(window=per.Y5).mean()

        df['prev_day_return'] = df['return'].shift(1)
        df['prev_day_bench_return'] = df['bench_return'].shift(1)

        df['ln_return'] = np.log(1 + df['prev_day_return'])
        df['ln_bench_return'] = np.log(1 + df['prev_day_bench_return'])

        df['weekly_return'] = np.exp(df['ln_return'].rolling(window=per.W).sum()) - 1
        df['monthly_return'] = np.exp(df['ln_return'].rolling(window=per.M).sum()) - 1
        df['annual_return'] = np.exp(df['ln_return'].rolling(window=per.Y).sum()) - 1
        df['3yr_return'] = np.exp(df['ln_return'].rolling(window=per.Y3).sum()) - 1
        df['5yr_return'] = np.exp(df['ln_return'].rolling(window=per.Y5).sum()) - 1

        df['weekly_bench_return'] = np.exp(df['ln_bench_return'].rolling(window=per.W).sum()) - 1
        df['monthly_bench_return'] = np.exp(df['ln_bench_return'].rolling(window=per.M).sum()) - 1
        df['annual_bench_return'] = np.exp(df['ln_bench_return'].rolling(window=per.Y).sum()) - 1
        df['3yr_bench_return'] = np.exp(df['ln_bench_return'].rolling(window=per.Y3).sum()) - 1
        df['5yr_bench_return'] = np.exp(df['ln_bench_return'].rolling(window=per.Y5).sum()) - 1

        df['monthly_volatility'] = df['prev_day_return'].rolling(window=per.M).std().values
        df['annual_volatility'] = df['prev_day_return'].rolling(window=per.Y).std().values

        df['monthly_bench_volatility'] = df['prev_day_bench_return'].rolling(window=per.M).std().values
        df['annual_bench_volatility'] = df['prev_day_bench_return'].rolling(window=per.Y).std().values

        dfs.append(df[
            ['ticker', 'date', 'prev_day_volume', 'weekly_volume', 'monthly_volume', 'annual_volume', '3yr_volume',
             '5yr_volume', 'prev_day_return', 'prev_day_bench_return', 'weekly_return',
             'weekly_bench_return', 'monthly_return', 'monthly_bench_return', 'annual_return',
             'annual_bench_return',
             '3yr_return', '3yr_bench_return', '5yr_return', '5yr_bench_return', 'monthly_volatility',
             'monthly_bench_volatility', 'annual_volatility', 'annual_bench_volatility', 'return']].fillna(
            0.0))

    return_df = pd.concat(dfs)
    return_df.set_index(['ticker', 'date'], inplace=True)
    return return_df
