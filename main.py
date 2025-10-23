"""
Main Execution Script - Pairs Trading with Kalman Filter
Run this file to execute the complete pipeline
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Import custom modules
from data_handler import DataHandler
from cointegration import CointegrationTester
from kalman_filter import KalmanHedgeRatio, KalmanOptimizer
from strategy import MeanReversionStrategy
from backtester import Backtester
from visualizer import Visualizer

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO",
    "TSLA", "BRK-B", "JPM", "V", "MA", "XOM", "CVX", "KO", "PEP",
    "PG", "C", "BAC", "WMT", "HD", "DIS", "NFLX", "ORCL", "JNJ",
    "PFE", "MRK", "INTC", "AMD"
]
START = "2009-01-01"
END = None  # Until today

# Strategy parameters
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
LOOKBACK = 20

# Cost parameters
COMMISSION_RATE = 0.00125  # 0.125%
BORROW_RATE = 0.0025  # 0.25% annualized
INITIAL_CAPITAL = 100000000
ALLOCATION_PCT = 0.8  # Use 80% of capital

# Kalman optimization
Q_RANGE = [1e-5, 1e-4, 1e-3, 1e-2]
R_RANGE = [0.01, 0.1, 1.0, 10.0]


def main():
    """Main execution pipeline"""

    print("=" * 80)
    print("PAIRS TRADING STRATEGY WITH KALMAN FILTER")
    print("Sequential Decision Process Framework")
    print("=" * 80)

    # =============================
    # 1. DOWNLOAD DATA
    # =============================
    print("\n[1] Downloading price data from Yahoo Finance...")
    print(f"    Tickers: {len(TICKERS)} assets")
    print(f"    Period: {START} to present")

    prices = yf.download(TICKERS, start=START, end=END)["Close"]
    prices = prices.dropna(how="any")

    print(f"    ✓ Downloaded {len(prices)} days of data")
    print(f"    Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # =============================
    # 2. CREATE DATA SPLITS
    # =============================
    print("\n[2] Creating train/test/validation splits...")
    data_handler = DataHandler(prices, train_pct=0.6, test_pct=0.2, val_pct=0.2)

    # =============================
    # 3. FIND COINTEGRATED PAIRS
    # =============================
    print("\n[3] Testing for cointegration on training data...")
    coint_tester = CointegrationTester(use_log=True, alpha=0.05)

    cointegrated_pairs = coint_tester.find_cointegrated_pairs(
        data_handler.train_data,
        min_corr=0.7
    )

    if len(cointegrated_pairs) == 0:
        print("\n❌ No cointegrated pairs found. Try:")
        print("   - Lowering min_corr threshold")
        print("   - Adding more tickers")
        print("   - Using different date range")
        return

    # Select best pair (lowest p-value = strongest cointegration)
    best_pair = sorted(cointegrated_pairs, key=lambda x: x['adf_p_residuals'])[0]
    asset1, asset2 = best_pair['asset1'], best_pair['asset2']

    print(f"\n✓ Found {len(cointegrated_pairs)} cointegrated pairs")
    print(f"✓ Selected best pair: {asset1} - {asset2}")
    print(f"  ADF p-value (residuals): {best_pair['adf_p_residuals']:.6f}")
    print(f"  Initial hedge ratio (OLS): {best_pair['beta1']:.4f}")
    print(f"  R-squared: {best_pair['r_squared']:.4f}")

    # =============================
    # 4. PREPARE PAIR DATA
    # =============================
    print("\n[4] Preparing pair data...")

    train_pair = data_handler.get_pair_data(asset1, asset2, 'train')
    test_pair = data_handler.get_pair_data(asset1, asset2, 'test')
    val_pair = data_handler.get_pair_data(asset1, asset2, 'val')

    # Use log prices (standard for cointegration)
    train_log = np.log(train_pair)
    test_log = np.log(test_pair)
    val_log = np.log(val_pair)

    # =============================
    # 5. OPTIMIZE KALMAN PARAMETERS
    # =============================
    print("\n[5] Optimizing Kalman Filter parameters (Q, R)...")
    print(f"    Grid search: {len(Q_RANGE)} Q values × {len(R_RANGE)} R values")

    optimizer = KalmanOptimizer(Q_range=Q_RANGE, R_range=R_RANGE)

    best_Q, best_R, grid_results = optimizer.optimize(
        train_log[asset1].values,
        train_log[asset2].values,
        verbose=True
    )

    # =============================
    # 6. RUN KALMAN FILTER ON ALL SPLITS
    # =============================
    print("\n[6] Running Kalman Filter with optimal parameters...")

    # Training set
    kf_train = KalmanHedgeRatio(Q=best_Q, R=best_R)
    train_kalman = kf_train.filter_series(
        train_log[asset1].values,
        train_log[asset2].values,
        beta_init=best_pair['beta1']
    )
    train_kalman.index = train_log.index

    # Test set
    kf_test = KalmanHedgeRatio(Q=best_Q, R=best_R)
    test_kalman = kf_test.filter_series(
        test_log[asset1].values,
        test_log[asset2].values,
        beta_init=best_pair['beta1']
    )
    test_kalman.index = test_log.index

    # Validation set
    kf_val = KalmanHedgeRatio(Q=best_Q, R=best_R)
    val_kalman = kf_val.filter_series(
        val_log[asset1].values,
        val_log[asset2].values,
        beta_init=best_pair['beta1']
    )
    val_kalman.index = val_log.index

    print(f"    ✓ Training:   β ∈ [{train_kalman['beta'].min():.4f}, {train_kalman['beta'].max():.4f}]")
    print(f"    ✓ Test:       β ∈ [{test_kalman['beta'].min():.4f}, {test_kalman['beta'].max():.4f}]")
    print(f"    ✓ Validation: β ∈ [{val_kalman['beta'].min():.4f}, {val_kalman['beta'].max():.4f}]")

    # =============================
    # 7. GENERATE TRADING SIGNALS
    # =============================
    print("\n[7] Generating trading signals...")

    strategy = MeanReversionStrategy(
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        lookback=LOOKBACK
    )

    train_signals = strategy.generate_signals(train_kalman['spread'].values)
    train_signals.index = train_kalman.index

    test_signals = strategy.generate_signals(test_kalman['spread'].values)
    test_signals.index = test_kalman.index

    val_signals = strategy.generate_signals(val_kalman['spread'].values)
    val_signals.index = val_kalman.index

    print(f"    Training:   {int((train_signals['signal'].diff() != 0).sum())} trades")
    print(f"    Test:       {int((test_signals['signal'].diff() != 0).sum())} trades")
    print(f"    Validation: {int((val_signals['signal'].diff() != 0).sum())} trades")

    # =============================
    # 8. CALCULATE POSITIONS
    # =============================
    print("\n[8] Calculating positions...")

    train_positions = strategy.calculate_positions(
        train_signals['signal'].values,
        train_kalman['beta'].values,
        train_pair[asset1].values,
        train_pair[asset2].values,
        INITIAL_CAPITAL,
        allocation_pct=ALLOCATION_PCT
    )
    train_positions.index = train_pair.index

    test_positions = strategy.calculate_positions(
        test_signals['signal'].values,
        test_kalman['beta'].values,
        test_pair[asset1].values,
        test_pair[asset2].values,
        INITIAL_CAPITAL,
        allocation_pct=ALLOCATION_PCT
    )
    test_positions.index = test_pair.index

    val_positions = strategy.calculate_positions(
        val_signals['signal'].values,
        val_kalman['beta'].values,
        val_pair[asset1].values,
        val_pair[asset2].values,
        INITIAL_CAPITAL,
        allocation_pct=ALLOCATION_PCT
    )
    val_positions.index = val_pair.index

    print(f"    ✓ Position sizing: ${INITIAL_CAPITAL:,.0f} capital, {ALLOCATION_PCT * 100:.0f}% allocation")

    # =============================
    # 9. BACKTEST WITH REALISTIC COSTS
    # =============================
    print("\n[9] Running backtests with transaction costs...")

    backtester = Backtester(
        commission_rate=COMMISSION_RATE,
        borrow_rate=BORROW_RATE,
        initial_capital=INITIAL_CAPITAL
    )

    # Backtest all splits
    train_results = backtester.run_backtest(train_positions, train_pair[asset1], train_pair[asset2])
    test_results = backtester.run_backtest(test_positions, test_pair[asset1], test_pair[asset2])
    val_results = backtester.run_backtest(val_positions, val_pair[asset1], val_pair[asset2])

    # Calculate metrics
    train_metrics = backtester.calculate_metrics(train_results)
    test_metrics = backtester.calculate_metrics(test_results)
    val_metrics = backtester.calculate_metrics(val_results)

    # Print results
    backtester.print_metrics(train_metrics, "TRAINING SET")
    backtester.print_metrics(test_metrics, "TEST SET")
    backtester.print_metrics(val_metrics, "VALIDATION SET")

    # =============================
    # 10. VISUALIZATIONS
    # =============================
    print("\n[10] Creating visualizations...")

    viz = Visualizer()

    # Plot 1: Spread Evolution (Test Set)
    fig1 = viz.plot_spread_evolution(
        test_kalman.index,
        test_kalman['spread'].values,
        test_signals['zscore'].values,
        test_signals['signal'].values,
        ENTRY_THRESHOLD,
        EXIT_THRESHOLD,
        title=f"Spread Evolution - {asset1}/{asset2} (Test Set)"
    )
    plt.savefig('spread_evolution.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: spread_evolution.png")

    # Plot 2: Hedge Ratio (Test Set)
    fig2 = viz.plot_hedge_ratio(
        test_kalman.index,
        test_kalman['beta'].values,
        test_kalman['P'].values,
        title=f"Dynamic Hedge Ratio - {asset1}/{asset2} (Test Set)"
    )
    plt.savefig('hedge_ratio.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: hedge_ratio.png")

    # Plot 3: Trade Returns Distribution (Test Set)
    if len(test_metrics['trade_returns']) > 0:
        fig3 = viz.plot_trade_returns(
            test_metrics['trade_returns'],
            title=f"Trade Returns Distribution - {asset1}/{asset2} (Test Set)"
        )
        plt.savefig('trade_returns.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved: trade_returns.png")
    else:
        print("    ⚠ No trades to plot distribution")

    # Plot 4: Portfolio Performance (Test Set)
    fig4 = viz.plot_portfolio_performance(
        test_results.index,
        test_results['portfolio_value'].values,
        INITIAL_CAPITAL,
        title=f"Portfolio Performance - {asset1}/{asset2} (Test Set)"
    )
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: portfolio_performance.png")

    # =============================
    # 11. FINAL SUMMARY
    # =============================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nPair: {asset1} - {asset2}")
    print(f"Cointegration p-value: {best_pair['adf_p_residuals']:.6f}")
    print(f"Optimal Kalman: Q={best_Q:.1e}, R={best_R:.2f}")
    print(f"\nOut-of-Sample Performance (Test Set):")
    print(f"  Sharpe Ratio:        {test_metrics['sharpe_ratio']:>8.2f}")
    print(f"  Total Return:        {test_metrics['total_return'] * 100:>7.2f}%")
    print(f"  Annualized Return:   {test_metrics['annualized_return'] * 100:>7.2f}%")
    print(f"  Max Drawdown:        {test_metrics['max_drawdown'] * 100:>7.2f}%")
    print(f"  Win Rate:            {test_metrics['win_rate'] * 100:>7.2f}%")
    print(f"  Number of Trades:    {test_metrics['num_trades']:>7}")
    print(f"  Final Value:         ${test_metrics['final_value']:>10,.2f}")

    print("\n" + "=" * 80)
    print("✓ EXECUTION COMPLETE")
    print("  - All plots saved as PNG files")
    print("  - Review results above")
    print("  - Check generated images")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()