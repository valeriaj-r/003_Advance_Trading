"""
Backtesting Module

Simulates strategy execution with realistic transaction costs:
- Commission: 0.125% per trade (entry and exit)
- Borrow cost: 0.25% annualized for short positions
"""
import numpy as np
import pandas as pd


class Backtester:
    """Backtests pairs trading strategy with realistic costs"""

    def __init__(self, commission_rate=0.00125, borrow_rate=0.0025, initial_capital=100000):
        """
        Args:
            commission_rate: Commission per trade (0.00125 = 0.125%)
            borrow_rate: Annualized borrow rate (0.0025 = 0.25%)
            initial_capital: Starting capital
        """
        self.commission_rate = commission_rate
        self.borrow_rate = borrow_rate
        self.initial_capital = initial_capital
        self.daily_borrow_rate = borrow_rate / 252  # Annualized to daily

    def calculate_trade_costs(self, position_changes, prices):
        """
        Calculate transaction costs for position changes

        Args:
            position_changes: Change in units for each asset
            prices: Current prices

        Returns:
            Total commission cost
        """
        trade_value = abs(position_changes * prices).sum()
        commission = trade_value * self.commission_rate
        return commission

    def calculate_borrow_costs(self, positions, prices):
        """
        Calculate daily borrow costs for short positions

        Args:
            positions: Current positions (negative = short)
            prices: Current prices

        Returns:
            Daily borrow cost
        """
        # Only pay borrow costs on short positions
        short_value = abs(positions[positions < 0] * prices[positions < 0]).sum()
        borrow_cost = short_value * self.daily_borrow_rate
        return borrow_cost

    def run_backtest(self, positions_df, prices_asset1, prices_asset2):
        """
        Execute backtest with full cost accounting

        Args:
            positions_df: DataFrame with columns: signal, units_asset1, units_asset2
            prices_asset1: Price series for asset 1
            prices_asset2: Price series for asset 2

        Returns:
            DataFrame with daily PnL, cumulative returns, trades
        """
        n = len(positions_df)

        # Initialize tracking arrays
        cash = np.zeros(n)
        portfolio_value = np.zeros(n)
        commissions = np.zeros(n)
        borrow_costs = np.zeros(n)
        trades = np.zeros(n)

        # Start with initial capital
        cash[0] = self.initial_capital

        prev_pos1 = 0
        prev_pos2 = 0

        for i in range(n):
            # Get current positions
            pos1 = positions_df.iloc[i]['units_asset1']
            pos2 = positions_df.iloc[i]['units_asset2']

            p1 = prices_asset1.iloc[i]
            p2 = prices_asset2.iloc[i]

            # Calculate position changes
            delta1 = pos1 - prev_pos1
            delta2 = pos2 - prev_pos2

            # Check if trade occurred
            if abs(delta1) > 1e-8 or abs(delta2) > 1e-8:
                trades[i] = 1

                # Calculate commission
                position_changes = pd.Series([delta1, delta2])
                current_prices = pd.Series([p1, p2])
                commission = self.calculate_trade_costs(position_changes, current_prices)
                commissions[i] = commission

            # Calculate borrow costs (daily)
            current_positions = pd.Series([pos1, pos2])
            current_prices = pd.Series([p1, p2])
            borrow_cost = self.calculate_borrow_costs(current_positions, current_prices)
            borrow_costs[i] = borrow_cost

            # Update cash
            if i > 0:
                # Carry forward previous cash
                cash[i] = cash[i - 1]

                # Add/subtract from position changes
                cash[i] -= delta1 * p1  # Buy = subtract, Sell = add
                cash[i] -= delta2 * p2

                # Subtract costs
                cash[i] -= commissions[i]
                cash[i] -= borrow_costs[i]

            # Calculate portfolio value
            position_value = pos1 * p1 + pos2 * p2
            portfolio_value[i] = cash[i] + position_value

            # Update previous positions
            prev_pos1 = pos1
            prev_pos2 = pos2

        # Calculate returns
        daily_returns = np.zeros(n)
        daily_returns[1:] = (portfolio_value[1:] - portfolio_value[:-1]) / portfolio_value[:-1]

        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_value,
            'cash': cash,
            'daily_return': daily_returns,
            'cumulative_return': (portfolio_value / self.initial_capital - 1),
            'commission': commissions,
            'borrow_cost': borrow_costs,
            'trade': trades,
            'position_asset1': positions_df['units_asset1'].values,
            'position_asset2': positions_df['units_asset2'].values
        }, index=positions_df.index)

        return results

    def calculate_metrics(self, backtest_results):
        """
        Calculate performance metrics

        Args:
            backtest_results: DataFrame from run_backtest

        Returns:
            Dictionary with performance metrics
        """
        returns = backtest_results['daily_return'].values
        portfolio_values = backtest_results['portfolio_value'].values

        # Remove first day (no return)
        returns = returns[1:]

        # Total return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital

        # Annualized return (252 trading days)
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = backtest_results['cumulative_return'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        max_drawdown = np.min(drawdown)

        # Win rate
        trades = backtest_results[backtest_results['trade'] == 1]
        if len(trades) > 1:
            # Calculate returns between trades
            trade_indices = trades.index.tolist()
            trade_returns = []
            for i in range(len(trade_indices) - 1):
                start_idx = backtest_results.index.get_loc(trade_indices[i])
                end_idx = backtest_results.index.get_loc(trade_indices[i + 1])
                trade_return = (portfolio_values[end_idx] - portfolio_values[start_idx]) / portfolio_values[start_idx]
                trade_returns.append(trade_return)

            trade_returns = np.array(trade_returns)
            win_rate = np.sum(trade_returns > 0) / len(trade_returns) if len(trade_returns) > 0 else 0
            avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
            avg_loss = np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            trade_returns = np.array([])

        # Total costs
        total_commissions = backtest_results['commission'].sum()
        total_borrow_costs = backtest_results['borrow_cost'].sum()

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': int(backtest_results['trade'].sum()),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_commissions': total_commissions,
            'total_borrow_costs': total_borrow_costs,
            'final_value': portfolio_values[-1],
            'trade_returns': trade_returns
        }

        return metrics

    def print_metrics(self, metrics, split_name=""):
        """Print performance metrics in a formatted way"""
        print(f"\n{'=' * 60}")
        print(f"Performance Metrics - {split_name}")
        print(f"{'=' * 60}")
        print(f"Total Return:          {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annualized Return:     {metrics['annualized_return'] * 100:>10.2f}%")
        print(f"Volatility (Ann.):     {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
        print(f"Max Drawdown:          {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Number of Trades:      {metrics['num_trades']:>10}")
        print(f"Win Rate:              {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Avg Win:               {metrics['avg_win'] * 100:>10.2f}%")
        print(f"Avg Loss:              {metrics['avg_loss'] * 100:>10.2f}%")
        print(f"Total Commissions:     ${metrics['total_commissions']:>10,.2f}")
        print(f"Total Borrow Costs:    ${metrics['total_borrow_costs']:>10,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_value']:>10,.2f}")
        print(f"{'=' * 60}\n")