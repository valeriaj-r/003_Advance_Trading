"""
Visualization module: Plot results
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_zscore_with_signals(dates, spread_history, trades, entry_threshold, exit_threshold,
                             ticker_A, ticker_B, window=30):
 
    # Calculate z-scores
    zscores = []
    for i in range(len(spread_history)):
        if i < window:
            zscores.append(0)
        else:
            recent_spreads = spread_history[max(0, i - window):i]
            mean = np.mean(recent_spreads)
            std = np.std(recent_spreads)
            if std > 1e-6:
                zscore = (spread_history[i] - mean) / std
            else:
                zscore = 0
            zscores.append(zscore)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot z-score
    ax.plot(dates, zscores, label='Z-Score', color='black', linewidth=1.5, alpha=0.7)

    # Plot threshold lines
    ax.axhline(y=entry_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Entry Threshold (+{entry_threshold})', alpha=0.7)
    ax.axhline(y=-entry_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Entry Threshold (-{entry_threshold})', alpha=0.7)
    ax.axhline(y=exit_threshold, color='green', linestyle='--', linewidth=1.5,
               label=f'Exit Threshold (+{exit_threshold})', alpha=0.5)
    ax.axhline(y=-exit_threshold, color='green', linestyle='--', linewidth=1.5,
               label=f'Exit Threshold (-{exit_threshold})', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.3)

    # Mark trade signals
    entry_long_dates = []
    entry_long_zscores = []
    entry_short_dates = []
    entry_short_zscores = []
    exit_dates = []
    exit_zscores = []

    for trade in trades:
        trade_date = trade['date']
        if trade_date not in dates:
            continue

        # Find index in dates
        try:
            idx = list(dates).index(trade_date)
            zscore_at_trade = zscores[idx]
        except (ValueError, IndexError):
            continue

        if trade['type'] == 'ENTRY':
            if trade['position'] == 'LONG_SPREAD':
                entry_long_dates.append(trade_date)
                entry_long_zscores.append(zscore_at_trade)
            else:  # SHORT_SPREAD
                entry_short_dates.append(trade_date)
                entry_short_zscores.append(zscore_at_trade)
        elif trade['type'] == 'EXIT':
            exit_dates.append(trade_date)
            exit_zscores.append(zscore_at_trade)

    # Plot signals
    if entry_long_dates:
        ax.scatter(entry_long_dates, entry_long_zscores, color='green', marker='^',
                   s=150, label='LONG Spread Entry', zorder=5, edgecolors='black', linewidths=1.5)
    if entry_short_dates:
        ax.scatter(entry_short_dates, entry_short_zscores, color='red', marker='v',
                   s=150, label='SHORT Spread Entry', zorder=5, edgecolors='black', linewidths=1.5)
    if exit_dates:
        ax.scatter(exit_dates, exit_zscores, color='orange', marker='x',
                   s=150, label='Exit Position', zorder=5, linewidths=2.5)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_title(f'Z-Score with Trading Signals: {ticker_A} - β×{ticker_B}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add shaded regions for entry zones
    ax.fill_between(dates, entry_threshold, 10, alpha=0.1, color='red',
                    label='_nolegend_')  # Short zone
    ax.fill_between(dates, -entry_threshold, -10, alpha=0.1, color='green',
                    label='_nolegend_')  # Long zone

    plt.tight_layout()
    return fig


def plot_spread_evolution(dates, spread_history, positions_history, ticker_A, ticker_B):
    """Plot spread evolution over time with trading positions"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot spread
    ax.plot(dates, spread_history, label='Spread', color='black', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Mark positions
    positions = np.array(positions_history)
    long_mask = positions == 1
    short_mask = positions == -1

    if np.any(long_mask):
        ax.fill_between(dates, spread_history, 0, where=long_mask,
                        alpha=0.3, color='green', label='Long Spread')
    if np.any(short_mask):
        ax.fill_between(dates, spread_history, 0, where=short_mask,
                        alpha=0.3, color='red', label='Short Spread')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Spread', fontsize=12)
    ax.set_title(f'Spread Evolution: {ticker_A} - β×{ticker_B}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_hedge_ratio(dates, beta_history, ticker_A, ticker_B):
    """Plot hedge ratio (beta) evolution over time"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, beta_history, label='Hedge Ratio (β)', color='blue', linewidth=1.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Hedge Ratio (β)', fontsize=12)
    ax.set_title(f'Dynamic Hedge Ratio: {ticker_A} vs {ticker_B}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig




def plot_portfolio_value(dates, portfolio_values, period_name, initial_capital):
    """Plot portfolio value over time"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, portfolio_values, label='Portfolio Value', color='darkblue', linewidth=2)
    ax.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5,
               label=f'Initial Capital: ${initial_capital:,.0f}')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title(f'Portfolio Value Evolution - {period_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    return fig


def print_performance_metrics(results, period_name):
    """Print performance metrics"""
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE METRICS - {period_name}")
    print(f"{'=' * 60}")
    print(f"Final Portfolio Value:  ${results['final_value']:,.2f}")
    print(f"Total Return:           {results['total_return'] * 100:.2f}%")
    print(f"Sharpe Ratio:           {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:           {results['max_drawdown'] * 100:.2f}%")
    print(f"Number of Trades:       {results['num_trades']}")

    if len(results['returns']) > 0:
        print(f"Mean Daily Return:      {np.mean(results['returns']) * 100:.4f}%")
        print(f"Std Daily Return:       {np.std(results['returns']) * 100:.4f}%")
        win_rate = np.sum(results['returns'] > 0) / len(results['returns']) * 100
        print(f"Win Rate (daily):       {win_rate:.2f}%")

    # Trade-level analysis
    if len(results['trades']) > 0:
        exit_trades = [t for t in results['trades'] if t['type'] == 'EXIT' and 'pnl_pct' in t]
        if exit_trades:
            trade_returns = [t['pnl_pct'] for t in exit_trades]
            winning_trades = sum(1 for r in trade_returns if r > 0)
            print(f"\nTrade-Level Stats:")
            print(
                f"  Winning Trades:       {winning_trades}/{len(trade_returns)} ({winning_trades / len(trade_returns) * 100:.1f}%)")
            print(f"  Avg Trade Return:     {np.mean(trade_returns) * 100:.2f}%")
            print(f"  Best Trade:           {np.max(trade_returns) * 100:.2f}%")
            print(f"  Worst Trade:          {np.min(trade_returns) * 100:.2f}%")

    print(f"{'=' * 60}\n")


def plot_trade_statistics(trades):
    """Plot trade-level statistics"""
    if len(trades) == 0:
        print("No trades to analyze")
        return None

    # Extract trade returns
    trade_returns = [t['pnl_pct'] for t in trades if t['type'] == 'EXIT']

    if len(trade_returns) == 0:
        print("No completed trades to analyze")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Trade returns distribution
    ax1.hist(trade_returns, bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(trade_returns), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(trade_returns) * 100:.2f}%')
    ax1.set_xlabel('Return per Trade (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Returns per Trade', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_results(results, data, ticker_A, ticker_B, period_name, initial_capital,
                      entry_threshold=None, exit_threshold=None):
    """
    Generate all visualizations for backtest results
    """
    if results is None:
        print(f"No results to visualize for {period_name}")
        return

    # Get dates - align with actual data length
    # The spread_history starts from index 0 
    start_idx = 30  # warm-up period
    end_idx = start_idx + len(results['portfolio_values'])
    dates = data.index[start_idx:end_idx]

    # Adjust spread and beta history to match portfolio values length
    spread_to_plot = results['spread_history'][-len(results['portfolio_values']):]
    beta_to_plot = results['beta_history'][-len(results['portfolio_values']):]
    positions_to_plot = results['positions_history']

    # Print metrics
    print_performance_metrics(results, period_name)

    # Plot Z-score with signals (NEW!)
    if entry_threshold is not None and exit_threshold is not None:
        plot_zscore_with_signals(dates, spread_to_plot, results['trades'],
                                 entry_threshold, exit_threshold, ticker_A, ticker_B)
        plt.show()

    # Plot spread evolution
    plot_spread_evolution(dates, spread_to_plot,
                          positions_to_plot, ticker_A, ticker_B)
    plt.show()

    # Plot hedge ratio
    plot_hedge_ratio(dates, beta_to_plot, ticker_A, ticker_B)
    plt.show()


    # Plot portfolio value
    plot_portfolio_value(dates, results['portfolio_values'], period_name, initial_capital)
    plt.show()

    # Plot trade statistics
    if len(results['trades']) > 0:
        fig = plot_trade_statistics(results['trades'])
        if fig:
            plt.show()


if __name__ == "__main__":
    print("Visualizations module loaded successfully")
