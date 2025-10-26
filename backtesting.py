"""
Backtesting module: Execute trading strategy with realistic costs
"""
import numpy as np
import pandas as pd
from kalman import KalmanFilterSDP, create_policy
from config import (INITIAL_CAPITAL, POSITION_SIZE_PCT,
                    COMMISSION_RATE, BORROW_RATE_DAILY)


class PairsTradingBacktest:
    """Backtest pairs trading strategy with Kalman Filter"""

    def __init__(self, ticker_A, ticker_B, initial_beta,
                 Q, R, entry_threshold, exit_threshold,
                 initial_capital=INITIAL_CAPITAL):
        """
        Initialize backtest

        Parameters:
        -----------
        ticker_A, ticker_B : str
            Asset pair tickers
        initial_beta : float
            Initial hedge ratio estimate
        Q, R : float
            Kalman filter parameters
        entry_threshold, exit_threshold : float
            Trading thresholds
        initial_capital : float
            Starting capital
        """
        self.ticker_A = ticker_A
        self.ticker_B = ticker_B
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # Kalman filter
        self.kf = KalmanFilterSDP(Q=Q, R=R)
        self.kf.initialize(initial_beta=initial_beta)

        # Policy
        self.policy = create_policy(entry_threshold, exit_threshold)

        # Position tracking
        self.position = 0  # 1 (long spread), -1 (short spread), 0 (no position)
        self.shares_A = 0
        self.shares_B = 0
        self.entry_value = 0

        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.trades = []
        self.positions_history = []

    def calculate_position_sizes(self, price_A, price_B, beta):
        """
        Calculate number of shares for each asset based on hedge ratio

        Uses 80% of available cash, split according to hedge ratio
        """
        available_capital = self.cash * POSITION_SIZE_PCT

        # Market neutral: dollar value of A = beta * dollar value of B
        # available_capital = price_A * shares_A + price_B * shares_B * beta
        # Solving: shares_A = available_capital / (2 * price_A)

        shares_A = available_capital / (2 * price_A)
        shares_B = (beta * shares_A * price_A) / price_B

        return shares_A, shares_B

    def calculate_commission(self, value):
        """Calculate trading commission"""
        return abs(value) * COMMISSION_RATE

    def calculate_borrow_cost(self, short_value):
        """Calculate daily borrow cost for short position"""
        return abs(short_value) * BORROW_RATE_DAILY

    def enter_position(self, date, price_A, price_B, position_type):
        """
        Enter new position

        position_type: 1 (long spread: long A, short B) - expect spread to increase
                      -1 (short spread: short A, long B) - expect spread to decrease
        """
        beta = self.kf.beta
        shares_A, shares_B = self.calculate_position_sizes(price_A, price_B, beta)

        if position_type == 1:  # Long spread: spread is LOW, expect it to INCREASE
            # Long A, Short B
            value_A = shares_A * price_A
            value_B = shares_B * price_B

            commission = self.calculate_commission(value_A) + self.calculate_commission(value_B)

            self.shares_A = shares_A
            self.shares_B = -shares_B  # Short
            self.cash -= value_A - value_B + commission
            self.entry_value = value_A + value_B  # Total position value

        elif position_type == -1:  # Short spread: spread is HIGH, expect it to DECREASE
            # Short A, Long B
            value_A = shares_A * price_A
            value_B = shares_B * price_B

            commission = self.calculate_commission(value_A) + self.calculate_commission(value_B)

            self.shares_A = -shares_A  # Short
            self.shares_B = shares_B
            self.cash += value_A - value_B - commission
            self.entry_value = value_A + value_B  # Total position value

        self.position = position_type

        self.trades.append({
            'date': date,
            'type': 'ENTRY',
            'position': 'LONG_SPREAD' if position_type == 1 else 'SHORT_SPREAD',
            'shares_A': self.shares_A,
            'shares_B': self.shares_B,
            'price_A': price_A,
            'price_B': price_B,
            'beta': beta,
            'commission': commission,
            'cash_after': self.cash
        })

    def exit_position(self, date, price_A, price_B):
        """Exit current position"""
        if self.position == 0:
            return

        # Current position values
        value_A = self.shares_A * price_A
        value_B = self.shares_B * price_B

        commission = self.calculate_commission(abs(value_A)) + self.calculate_commission(abs(value_B))

        # Calculate P&L
        current_position_value = abs(value_A) + abs(value_B)

        # Close position - reverse the trades
        pnl = 0
        if self.position == 1:  # Was long spread (long A, short B)
            # Close: sell A, cover B
            pnl = value_A + value_B  # Profit from long A, cost to cover short B
        elif self.position == -1:  # Was short spread (short A, long B)
            # Close: cover A, sell B
            pnl = value_A + value_B  # Cost to cover short A, profit from long B

        self.cash += pnl - commission

        # Calculate percentage return on this trade
        pnl_pct = (current_position_value - self.entry_value) / self.entry_value if self.entry_value > 0 else 0

        # If short spread, invert the PnL logic (profit when spread decreases)
        if self.position == -1:
            pnl_pct = -pnl_pct

        self.trades.append({
            'date': date,
            'type': 'EXIT',
            'position': 'LONG_SPREAD' if self.position == 1 else 'SHORT_SPREAD',
            'shares_A': self.shares_A,
            'shares_B': self.shares_B,
            'price_A': price_A,
            'price_B': price_B,
            'beta': self.kf.beta,
            'commission': commission,
            'pnl_pct': pnl_pct,
            'cash_after': self.cash
        })

        self.shares_A = 0
        self.shares_B = 0
        self.position = 0
        self.entry_value = 0

    def update_portfolio_value(self, price_A, price_B):
        """Calculate current portfolio value"""
        position_value_A = self.shares_A * price_A
        position_value_B = self.shares_B * price_B

        # Subtract daily borrow costs for short positions
        borrow_cost = 0
        if self.shares_A < 0:
            borrow_cost += self.calculate_borrow_cost(position_value_A)
        if self.shares_B < 0:
            borrow_cost += self.calculate_borrow_cost(position_value_B)

        self.cash -= borrow_cost

        total_value = self.cash + position_value_A + position_value_B
        return total_value

    def run(self, data, start_idx=30):
        """
        Run backtest on data

        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns [ticker_A, ticker_B]
        start_idx : int
            Start trading after this many days (for z-score window)
        """
        prices_A = data[self.ticker_A].values
        prices_B = data[self.ticker_B].values
        dates = data.index

        # Initialize
        log_prices_A = np.log(prices_A)
        log_prices_B = np.log(prices_B)

        # Warm-up Kalman filter
        for i in range(1, start_idx):
            self.kf.predict()
            self.kf.update(log_prices_A[i], log_prices_B[i])
            self.kf.compute_spread(log_prices_A[i], log_prices_B[i])

        # Main trading loop
        for i in range(start_idx, len(data)):
            # Kalman update (exogenous information Wt+1)
            self.kf.predict()
            self.kf.update(log_prices_A[i], log_prices_B[i])
            spread = self.kf.compute_spread(log_prices_A[i], log_prices_B[i])

            # Policy decision
            zscore = self.kf.compute_zscore(window=30)
            decision = self.policy(zscore, self.position)

            # Execute decision
            if decision != self.position:
                if self.position != 0:
                    self.exit_position(dates[i], prices_A[i], prices_B[i])

                if decision != 0:
                    self.enter_position(dates[i], prices_A[i], prices_B[i], decision)

            # Update portfolio value
            portfolio_value = self.update_portfolio_value(prices_A[i], prices_B[i])
            self.portfolio_values.append(portfolio_value)
            self.positions_history.append(self.position)

            # Calculate returns
            if len(self.portfolio_values) > 1:
                ret = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
                self.returns.append(ret)

        # Close any open position at end
        if self.position != 0:
            self.exit_position(dates[-1], prices_A[-1], prices_B[-1])

        return self.get_results()

    def get_results(self):
        """Get backtest results"""
        if len(self.portfolio_values) == 0:
            return None

        final_value = self.portfolio_values[-1]
        total_return = (final_value / self.initial_capital) - 1

        returns_array = np.array(self.returns) if self.returns else np.array([0])

        # Ensure all histories have the same length as portfolio_values
        n = len(self.portfolio_values)
        spread_history = self.kf.spread_history[-n:] if len(self.kf.spread_history) >= n else self.kf.spread_history
        beta_history = self.kf.beta_history[-n:] if len(self.kf.beta_history) >= n else self.kf.beta_history

        results = {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(
                returns_array) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(),
            'num_trades': len([t for t in self.trades if t['type'] == 'ENTRY']),
            'portfolio_values': self.portfolio_values,
            'returns': returns_array,
            'trades': self.trades,
            'beta_history': beta_history,
            'spread_history': spread_history,
            'positions_history': self.positions_history
        }

        return results

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) == 0:
            return 0

        values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return np.min(drawdown)


if __name__ == "__main__":
    print("Backtesting module loaded successfully")

