"""
Data module: Download and split data
"""
import pandas as pd
import yfinance as yf
from config import START_DATE, END_DATE, TRAIN_RATIO, TEST_RATIO, SECTORS


def download_data(tickers, start_date=START_DATE, end_date=END_DATE):
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Handle missing data
    data = data.dropna(axis=1, how='all')  # Remove columns with all NaN
    data = data.fillna(method='ffill').fillna(method='bfill')  # Forward/backward fill, better than ignore those days

    print(f"Downloaded {len(data.columns)} valid tickers with {len(data)} days of data")
    return data


def split_data(data, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO):
    """Split data chronologically into train, test, validation"""
    n = len(data)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))

    train = data.iloc[:train_end]
    test = data.iloc[train_end:test_end]
    validation = data.iloc[test_end:]

    print(f"Data split - Train: {len(train)} days, Test: {len(test)} days, Validation: {len(validation)} days")
    return train, test, validation


def get_all_tickers():
    """Get all unique tickers from sectors"""
    all_tickers = []
    for sector, tickers in SECTORS.items():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


if __name__ == "__main__":
    # Test data download
    tickers = get_all_tickers()
    data = download_data(tickers)
    train, test, val = split_data(data)
    print(f"\nTrain period: {train.index[0]} to {train.index[-1]}")
    print(f"Test period: {test.index[0]} to {test.index[-1]}")
    print(f"Validation period: {val.index[0]} to {val.index[-1]}")
