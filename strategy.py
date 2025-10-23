"""
Trading Strategy Module

Policy π(S_t) that generates trading decisions based on:
- Spread Z-score (normalized deviation from mean)
- Dynamic hedge ratio from Kalman filter
"""
import numpy as np
import pandas as pd


class MeanReversionStrategy:
    """
    Mean reversion trading strategy using Z-score signals

    Policy: X^π(S_t) where:
        - S_t = (spread, z_score, position_state)
        - X_t = position vector (units of asset1, asset2)
    """

    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, lookback=20):
        """
        Args:
            entry_threshold: |Z-score| threshold to enter trade
            exit_threshold: |Z-score| threshold to exit trade
            lookback: Window for calculating spread statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback

    def calculate_zscore(self, spreads):
        """
        Calculate rolling Z-score of spread

        Z_t = (spread_t - μ) / σ
        where μ and σ are rolling statistics
        """
        spread_series = pd.Series(spreads)

        rolling_mean = spread_series.rolling(window=self.lookback, min_periods=1).mean()
        rolling_std = spread_series.rolling(window=self.lookback, min_periods=1).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)

        zscore = (spread_series - rolling_mean) / rolling_std

        return zscore.values, rolling_mean.values, rolling_std.values

    def generate_signals(self, spreads):
        """
        Generate trading signals based on Z-score

        Returns:
            DataFrame with signals:
                1: Long spread (buy asset1, short asset2)
                -1: Short spread (short asset1, buy asset2)
                0: No position / exit
        """
        zscores, means, stds = self.calculate_zscore(spreads)

        signals = np.zeros(len(spreads))
        position = 0  # Current position state

        for i in range(len(spreads)):
            z = zscores[i]

            if position == 0:
                # No position - check for entry
                if z > self.entry_threshold:
                    # Spread too high → short spread
                    signals[i] = -1
                    position = -1
                elif z < -self.entry_threshold:
                    # Spread too low → long spread
                    signals[i] = 1
                    position = 1

            elif position == 1:
                # Long spread position
                if abs(z) < self.exit_threshold:
                    # Exit condition
                    signals[i] = 0
                    position = 0
                else:
                    # Hold position
                    signals[i] = 1

            elif position == -1:
                # Short spread position
                if abs(z) < self.exit_threshold:
                    # Exit condition
                    signals[i] = 0
                    position = 0
                else:
                    # Hold position
                    signals[i] = -1

        return pd.DataFrame({
            'signal': signals,
            'zscore': zscores,
            'spread_mean': means,
            'spread_std': stds
        })

    def calculate_positions(self, signals, hedge_ratios, prices_asset1, prices_asset2,
                            capital, allocation_pct=0.8):
        """
        Calculate actual positions (units of each asset) based on signals

        Position sizing: Allocate capital according to hedge ratio

        Args:
            signals: Trading signals (-1, 0, 1)
            hedge_ratios: Dynamic hedge ratios from Kalman filter
            prices_asset1: Prices of first asset
            prices_asset2: Prices of second asset
            capital: Available capital
            allocation_pct: Percentage of capital to use (0.8 = 80%)

        Returns:
            DataFrame with positions
        """
        positions = pd.DataFrame({
            'signal': signals,
            'hedge_ratio': hedge_ratios,
            'price1': prices_asset1,
            'price2': prices_asset2
        })

        # Calculate position sizes
        positions['units_asset1'] = 0.0
        positions['units_asset2'] = 0.0

        for i in range(len(positions)):
            signal = positions.loc[i, 'signal']

            if signal == 0:
                # No position
                continue

            # Get hedge ratio and prices
            beta = positions.loc[i, 'hedge_ratio']
            p1 = positions.loc[i, 'price1']
            p2 = positions.loc[i, 'price2']

            # Allocate capital
            trade_capital = capital * allocation_pct

            # Position sizing based on hedge ratio
            # Total value: V1 + β*V2 where V1 = units1 * p1, V2 = units2 * p2
            # We want: V1 = β * V2 (according to hedge ratio)

            if signal == 1:
                # Long spread: long asset1, short asset2
                # Allocate proportionally: more capital to asset with higher weight
                value_asset2 = trade_capital / (1 + beta)
                value_asset1 = trade_capital - value_asset2

                positions.loc[i, 'units_asset1'] = value_asset1 / p1
                positions.loc[i, 'units_asset2'] = -value_asset2 / p2

            elif signal == -1:
                # Short spread: short asset1, long asset2
                value_asset2 = trade_capital / (1 + beta)
                value_asset1 = trade_capital - value_asset2

                positions.loc[i, 'units_asset1'] = -value_asset1 / p1
                positions.loc[i, 'units_asset2'] = value_asset2 / p2

        return positions