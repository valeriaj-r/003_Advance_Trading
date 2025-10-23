"""
Visualization Module
Creates required plots for analysis
"""
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Creates analytical plots for strategy performance"""

    @staticmethod
    def plot_spread_evolution(dates, spreads, zscores, signals, entry_thresh, exit_thresh,
                              title="Spread Evolution"):
        """
        Plot spread and Z-score with trading signals

        Args:
            dates: Date index
            spreads: Spread values
            zscores: Z-score values
            signals: Trading signals
            entry_thresh: Entry threshold
            exit_thresh: Exit threshold
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Plot spread
        ax1.plot(dates, spreads, label='Spread', color='blue', linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_ylabel('Spread', fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot Z-score with thresholds
        ax2.plot(dates, zscores, label='Z-score', color='purple', linewidth=1)
        ax2.axhline(y=entry_thresh, color='red', linestyle='--', linewidth=1,
                    label=f'Entry threshold (±{entry_thresh})')
        ax2.axhline(y=-entry_thresh, color='red', linestyle='--', linewidth=1)
        ax2.axhline(y=exit_thresh, color='green', linestyle='--', linewidth=1,
                    label=f'Exit threshold (±{exit_thresh})')
        ax2.axhline(y=-exit_thresh, color='green', linestyle='--', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Mark trade entries
        entries = np.where(np.diff(signals.astype(float)) != 0)[0]
        if len(entries) > 0:
            ax2.scatter(dates[entries], zscores[entries], color='orange',
                        s=50, zorder=5, label='Trade entry/exit', marker='o')

        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Z-score', fontsize=11)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_hedge_ratio(dates, hedge_ratios, uncertainty=None, title="Dynamic Hedge Ratio"):
        """
        Plot hedge ratio evolution over time

        Args:
            dates: Date index
            hedge_ratios: Hedge ratio values
            uncertainty: Optional uncertainty bands (P from Kalman)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(dates, hedge_ratios, label='Hedge Ratio (β)', color='darkblue', linewidth=1.5)

        # Add uncertainty bands if available
        if uncertainty is not None:
            std = np.sqrt(uncertainty)
            ax.fill_between(dates,
                            hedge_ratios - 2 * std,
                            hedge_ratios + 2 * std,
                            alpha=0.2, color='blue',
                            label='95% Confidence interval')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Hedge Ratio (β)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_trade_returns(trade_returns, title="Distribution of Returns per Trade"):
        """
        Plot histogram of trade returns

        Args:
            trade_returns: Array of returns per trade
            title: Plot title
        """
        if len(trade_returns) == 0:
            print("No trades to plot")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(trade_returns * 100, bins=30, color='steelblue',
                 edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=np.mean(trade_returns) * 100, color='green',
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(trade_returns) * 100:.2f}%')
        ax1.set_xlabel('Return (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Trade Returns Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_returns = np.sort(trade_returns * 100)
        cumulative = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        ax2.plot(sorted_returns, cumulative, color='darkblue', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.set_xlabel('Return (%)', fontsize=11)
        ax2.set_ylabel('Cumulative Probability', fontsize=11)
        ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_portfolio_performance(dates, portfolio_values, initial_capital,
                                   title="Portfolio Performance"):
        """
        Plot portfolio value evolution

        Args:
            dates: Date index
            portfolio_values: Portfolio value over time
            initial_capital: Starting capital
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        returns = (portfolio_values - initial_capital) / initial_capital * 100

        ax.plot(dates, portfolio_values, label='Portfolio Value',
                color='darkgreen', linewidth=2)
        ax.axhline(y=initial_capital, color='black', linestyle='--',
                   linewidth=1, label='Initial Capital')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add text with final return
        final_return = returns.iloc[-1] if hasattr(returns, 'iloc') else returns[-1]
        ax.text(0.02, 0.98, f'Total Return: {final_return:.2f}%',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig