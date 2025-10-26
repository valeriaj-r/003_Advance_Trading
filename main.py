"""
Main module: Orchestrate the entire pairs trading strategy
"""

import numpy as np
import pandas as pd
from itertools import product

from data import download_data, split_data, get_all_tickers
from cointegration import find_all_cointegrated_pairs
from backtesting import PairsTradingBacktest
from visualizations import visualize_results
from config import (Q_VALUES, R_VALUES, ENTRY_THRESHOLDS, EXIT_THRESHOLDS,
                    INITIAL_CAPITAL)


def optimize_parameters(train_data, ticker_A, ticker_B, initial_beta):
    """
    Step 5: DESIGNING POLICIES
    Grid search over Kalman and trading parameters

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    ticker_A, ticker_B : str
        Asset pair
    initial_beta : float
        Initial hedge ratio estimate

    Returns:
    --------
    dict with best parameters and performance
    """
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZING PARAMETERS FOR {ticker_A} - {ticker_B}")
    print(f"{'=' * 60}")
    print(f"Testing {len(Q_VALUES)} Q values × {len(R_VALUES)} R values × "
          f"{len(ENTRY_THRESHOLDS)} entry × {len(EXIT_THRESHOLDS)} exit = "
          f"{len(Q_VALUES) * len(R_VALUES) * len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)} combinations")

    best_sharpe = -np.inf
    best_params = None
    best_results = None

    # Grid search
    total_combinations = len(Q_VALUES) * len(R_VALUES) * len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)
    counter = 0

    for Q in Q_VALUES:
        for R in R_VALUES:
            for entry_th in ENTRY_THRESHOLDS:
                for exit_th in EXIT_THRESHOLDS:
                    counter += 1

                    # Run backtest
                    backtest = PairsTradingBacktest(
                        ticker_A=ticker_A,
                        ticker_B=ticker_B,
                        initial_beta=initial_beta,
                        Q=Q,
                        R=R,
                        entry_threshold=entry_th,
                        exit_threshold=exit_th,
                        initial_capital=INITIAL_CAPITAL
                    )

                    results = backtest.run(train_data)

                    if results and results['sharpe_ratio'] > best_sharpe:
                        best_sharpe = results['sharpe_ratio']
                        best_params = {
                            'Q': Q,
                            'R': R,
                            'entry_threshold': entry_th,
                            'exit_threshold': exit_th
                        }
                        best_results = results

                    # Progress update
                    if counter % 10 == 0 or counter == total_combinations:
                        print(f"Progress: {counter}/{total_combinations} "
                              f"({counter / total_combinations * 100:.1f}%) - "
                              f"Best Sharpe: {best_sharpe:.3f}")

    print(f"\n{'=' * 60}")
    print(f"BEST PARAMETERS FOUND:")
    print(f"{'=' * 60}")
    for key, value in best_params.items():
        print(f"{key:20s}: {value}")
    print(f"{'=' * 60}\n")

    return best_params, best_results


def run_strategy_for_pair(train_data, test_data, val_data,
                          ticker_A, ticker_B, initial_beta):
    """
    Run complete strategy for a single pair:
    1. Optimize on train
    2. Evaluate on test
    3. Validate on validation
    """
    print(f"\n{'#' * 70}")
    print(f"RUNNING STRATEGY: {ticker_A} - {ticker_B}")
    print(f"{'#' * 70}")

    # Step 5: Optimize parameters on training data
    best_params, train_results = optimize_parameters(
        train_data, ticker_A, ticker_B, initial_beta
    )

    # Step 6: EVALUATING POLICIES on Test set
    print(f"\n{'=' * 60}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'=' * 60}")

    test_backtest = PairsTradingBacktest(
        ticker_A=ticker_A,
        ticker_B=ticker_B,
        initial_beta=initial_beta,
        Q=best_params['Q'],
        R=best_params['R'],
        entry_threshold=best_params['entry_threshold'],
        exit_threshold=best_params['exit_threshold'],
        initial_capital=INITIAL_CAPITAL
    )
    test_results = test_backtest.run(test_data)

    # Step 6: EVALUATING POLICIES on Validation set
    print(f"\n{'=' * 60}")
    print(f"EVALUATING ON VALIDATION SET")
    print(f"{'=' * 60}")

    val_backtest = PairsTradingBacktest(
        ticker_A=ticker_A,
        ticker_B=ticker_B,
        initial_beta=initial_beta,
        Q=best_params['Q'],
        R=best_params['R'],
        entry_threshold=best_params['entry_threshold'],
        exit_threshold=best_params['exit_threshold'],
        initial_capital=INITIAL_CAPITAL
    )
    val_results = val_backtest.run(val_data)

    # Visualize results
    print(f"\n{'=' * 60}")
    print(f"VISUALIZING RESULTS")
    print(f"{'=' * 60}")

    visualize_results(train_results, train_data, ticker_A, ticker_B,
                      "TRAIN", INITIAL_CAPITAL,
                      entry_threshold=best_params['entry_threshold'],
                      exit_threshold=best_params['exit_threshold'])
    visualize_results(test_results, test_data, ticker_A, ticker_B,
                      "TEST", INITIAL_CAPITAL,
                      entry_threshold=best_params['entry_threshold'],
                      exit_threshold=best_params['exit_threshold'])
    visualize_results(val_results, val_data, ticker_A, ticker_B,
                      "VALIDATION", INITIAL_CAPITAL,
                      entry_threshold=best_params['entry_threshold'],
                      exit_threshold=best_params['exit_threshold'])

    return {
        'params': best_params,
        'train': train_results,
        'test': test_results,
        'validation': val_results
    }


def main():
    """
    Main execution following Powell's 6-step process
    """
    print("=" * 70)
    print("PAIRS TRADING WITH KALMAN FILTER")
    print("Sequential Decision Process Framework (Powell)")
    print("=" * 70)

    # Step 1: THE NARRATIVE (see docstring above)
    # Step 2: CORE ELEMENTS (see docstring above)

    # Step 3: MATHEMATICAL MODEL - Download and prepare data
    print("\n[STEP 3: MATHEMATICAL MODEL - Data Preparation]")
    tickers = get_all_tickers()
    full_data = download_data(tickers)
    train_data, test_data, val_data = split_data(full_data)

    # Step 3: Find cointegrated pairs
    print("\n[STEP 3: MATHEMATICAL MODEL - Cointegration Testing]")
    cointegrated_pairs = find_all_cointegrated_pairs(train_data)

    if not cointegrated_pairs:
        print("\nNo cointegrated pairs found. Exiting.")
        return

    # Select best pair (lowest p-value) from first sector with pairs
    best_pair = None
    best_sector = None
    best_pvalue = 1.0

    for sector, pairs in cointegrated_pairs.items():
        for ticker_A, ticker_B, results in pairs:
            if results['adf_p_residuals'] < best_pvalue:
                best_pvalue = results['adf_p_residuals']
                best_pair = (ticker_A, ticker_B, results)
                best_sector = sector

    if best_pair is None:
        print("\nCould not select best pair. Exiting.")
        return

    ticker_A, ticker_B, coint_results = best_pair
    initial_beta = coint_results['beta']

    print(f"\n{'=' * 60}")
    print(f"SELECTED PAIR: {ticker_A} - {ticker_B}")
    print(f"Sector: {best_sector}")
    print(f"ADF p-value (residuals): {best_pvalue:.6f}")
    print(f"Initial Beta (OLS): {initial_beta:.4f}")
    print(f"{'=' * 60}")

    # Prepare pair data
    pair_train = train_data[[ticker_A, ticker_B]].dropna()
    pair_test = test_data[[ticker_A, ticker_B]].dropna()
    pair_val = val_data[[ticker_A, ticker_B]].dropna()

    # Step 4: UNCERTAINTY MODEL (handled in Kalman filter and grid search)
    # Step 5 & 6: DESIGNING and EVALUATING POLICIES
    results = run_strategy_for_pair(
        pair_train, pair_test, pair_val,
        ticker_A, ticker_B, initial_beta
    )

    # Final summary
    print(f"\n{'#' * 70}")
    print(f"FINAL SUMMARY")
    print(f"{'#' * 70}")
    print(f"Pair: {ticker_A} - {ticker_B}")
    print(f"Optimal Parameters:")
    for key, value in results['params'].items():
        print(f"  {key}: {value}")

    print(f"\nPerformance Summary:")
    for period in ['train', 'test', 'validation']:
        if results[period]:
            print(f"\n{period.upper()}:")
            print(f"  Total Return: {results[period]['total_return'] * 100:.2f}%")
            print(f"  Sharpe Ratio: {results[period]['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {results[period]['max_drawdown'] * 100:.2f}%")
            print(f"  Num Trades: {results[period]['num_trades']}")

    print(f"\n{'#' * 70}")
    print("STRATEGY EXECUTION COMPLETE")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()