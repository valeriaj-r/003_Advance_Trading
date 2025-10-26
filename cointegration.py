"""
Cointegration module: Find cointegrated pairs within sectors
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from config import COINTEGRATION_ALPHA, USE_LOG_PRICES, SECTORS


def test_cointegration(s1, s2, use_log=USE_LOG_PRICES, alpha=COINTEGRATION_ALPHA):
    """
    Test if two series are cointegrated using Engle-Granger method
    Returns dict with test results and OLS parameters
    """
    # Use log prices if specified
    if use_log:
        s1 = np.log(s1)
        s2 = np.log(s2)

    # Test 1: Both series should be non-stationary (ADF p > alpha)
    p1 = adfuller(s1.values, regression="c", autolag="AIC")[1]
    p2 = adfuller(s2.values, regression="c", autolag="AIC")[1]

    # Test 2: OLS regression and test residuals for stationarity
    X = sm.add_constant(s2.values)
    model = sm.OLS(s1.values, X).fit()
    w0, w1 = model.params
    residuals = s1.values - (w0 + w1 * s2.values)

    # ADF test on residuals (should be stationary, p < alpha)
    p_res = adfuller(residuals, regression="n", autolag="AIC")[1]

    # Check cointegration conditions
    cond_nonstat = (p1 > alpha) and (p2 > alpha)
    cond_res_stat = (p_res < alpha)
    is_cointegrated = cond_nonstat and cond_res_stat

    return {
        "is_cointegrated": is_cointegrated,
        "adf_p_s1": p1,
        "adf_p_s2": p2,
        "adf_p_residuals": p_res,
        "beta": w1,
        "intercept": w0
    }


def find_cointegrated_pairs(data, sector_name, tickers):
    """
    Find all cointegrated pairs within a sector
    Returns list of (ticker1, ticker2, test_results) tuples
    """
    cointegrated_pairs = []

    # Test all possible pairs
    pairs = list(combinations(tickers, 2))
    print(f"\nTesting {len(pairs)} pairs in {sector_name}...")

    for ticker1, ticker2 in pairs:
        if ticker1 not in data.columns or ticker2 not in data.columns:
            continue

        df_pair = data[[ticker1, ticker2]].dropna()
        if len(df_pair) < 100:  # Need sufficient data
            continue

        result = test_cointegration(df_pair[ticker1], df_pair[ticker2])

        if result["is_cointegrated"]:
            cointegrated_pairs.append((ticker1, ticker2, result))
            print(f"  ✓ Found: {ticker1} - {ticker2} (p-value: {result['adf_p_residuals']:.4f})")

    return cointegrated_pairs


def find_all_cointegrated_pairs(data):
    """
    Find cointegrated pairs across all sectors
    Returns dict: {sector: [(ticker1, ticker2, results), ...]}
    """
    all_pairs = {}

    for sector, tickers in SECTORS.items():
        pairs = find_cointegrated_pairs(data, sector, tickers)
        if pairs:
            all_pairs[sector] = pairs

    total = sum(len(pairs) for pairs in all_pairs.values())
    print(f"\n{'=' * 60}")
    print(f"Total cointegrated pairs found: {total}")
    print(f"{'=' * 60}")

    return all_pairs


if __name__ == "__main__":
    from data import download_data, get_all_tickers

    tickers = get_all_tickers()
    data = download_data(tickers)
    pairs = find_all_cointegrated_pairs(data)

    for sector, pair_list in pairs.items():
        print(f"\n{sector}: {len(pair_list)} pairs")