"""
Kalman Filter Module (Sequential Decision Process Framework)

Following Powell's Sequential Decision Analytics:
- State (S_t): Current estimate of hedge ratio β and uncertainty P
- Decision: Update belief state using new observations
- Exogenous Info (W_t+1): New price observations
- Transition: Kalman predict + update equations
- Objective: Minimize estimation error variance
"""
import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    """
    Kalman Filter for dynamic hedge ratio estimation

    State space model:
        β_t = β_{t-1} + w_t,  w_t ~ N(0, Q)  [State equation]
        y_t = β_t * x_t + v_t,  v_t ~ N(0, R)  [Observation equation]

    Where:
        β_t: hedge ratio at time t (our state variable)
        y_t: price of asset 1 (what we observe)
        x_t: price of asset 2 (what we observe)
        Q: process noise variance (how much β can change)
        R: measurement noise variance (observation uncertainty)
    """

    def __init__(self, Q=1e-4, R=1.0):
        """
        Args:
            Q: Process noise variance (state evolution)
            R: Measurement noise variance (observation noise)
        """
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise

        # State variables (S_t in Powell's framework)
        self.beta = None  # Current estimate of hedge ratio
        self.P = None  # Uncertainty in estimate (covariance)

        # History tracking
        self.beta_history = []
        self.P_history = []
        self.spread_history = []

    def initialize(self, beta_init, P_init=1.0):
        """
        Initialize state (S_0 in Powell's framework)

        Args:
            beta_init: Initial hedge ratio estimate (from OLS)
            P_init: Initial uncertainty
        """
        self.beta = beta_init
        self.P = P_init
        self.beta_history = [beta_init]
        self.P_history = [P_init]
        self.spread_history = []

    def predict(self):
        """
        Prediction step (time update)
        State transition: S_{t+1} = S_M(S_t, x_t, W_{t+1})

        Our model: β_{t|t-1} = β_{t-1|t-1} (random walk)
        """
        # State prediction (no change expected in random walk)
        beta_pred = self.beta

        # Uncertainty increases due to process noise
        P_pred = self.P + self.Q

        return beta_pred, P_pred

    def update(self, y_t, x_t):
        """
        Update step (measurement update)
        Incorporate new exogenous information W_{t+1} = (y_t, x_t)

        Args:
            y_t: Observed price of asset 1
            x_t: Observed price of asset 2
        """
        # Prediction
        beta_pred, P_pred = self.predict()

        # Innovation (prediction error)
        y_pred = beta_pred * x_t
        innovation = y_t - y_pred

        # Innovation variance
        S = x_t ** 2 * P_pred + self.R

        # Kalman gain (optimal weighting)
        K = (P_pred * x_t) / S

        # State update
        self.beta = beta_pred + K * innovation
        self.P = (1 - K * x_t) * P_pred

        # Calculate spread
        spread = y_t - self.beta * x_t

        # Store history
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.spread_history.append(spread)

        return self.beta, spread

    def filter_series(self, y_series, x_series, beta_init=None):
        """
        Run Kalman filter on entire time series

        Args:
            y_series: Price series of asset 1 (dependent)
            x_series: Price series of asset 2 (independent)
            beta_init: Initial hedge ratio (if None, uses OLS)

        Returns:
            DataFrame with hedge ratios and spreads
        """
        if beta_init is None:
            # Initialize with OLS estimate
            beta_init = np.cov(y_series, x_series)[0, 1] / np.var(x_series)

        self.initialize(beta_init)

        betas = []
        spreads = []

        for y_t, x_t in zip(y_series, x_series):
            beta, spread = self.update(y_t, x_t)
            betas.append(beta)
            spreads.append(spread)

        return pd.DataFrame({
            'beta': betas,
            'spread': spreads,
            'P': self.P_history[1:]  # Skip initial
        })


class KalmanOptimizer:
    """Grid search for optimal Q and R parameters"""

    def __init__(self, Q_range=None, R_range=None):
        """
        Args:
            Q_range: List of Q values to test
            R_range: List of R values to test
        """
        self.Q_range = Q_range if Q_range else [1e-5, 1e-4, 1e-3, 1e-2]
        self.R_range = R_range if R_range else [0.01, 0.1, 1.0, 10.0]

    def _evaluate_params(self, y_train, x_train, Q, R):
        """
        Evaluate a specific (Q, R) combination

        Metric: Negative spread variance (want stable, mean-reverting spread)
        """
        kf = KalmanHedgeRatio(Q=Q, R=R)
        result = kf.filter_series(y_train, x_train)

        spread = result['spread'].values

        # Metric: Combination of spread stationarity and stability
        spread_std = np.std(spread)
        spread_mean = np.abs(np.mean(spread))

        # Lower is better (stable spread around zero)
        score = spread_std + spread_mean

        return score

    def optimize(self, y_train, x_train, verbose=True):
        """
        Find optimal Q and R via grid search

        Args:
            y_train: Training prices of asset 1
            x_train: Training prices of asset 2
            verbose: Print progress

        Returns:
            Best Q, best R, and all results
        """
        results = []

        if verbose:
            print(f"Grid search: {len(self.Q_range)} Q values × {len(self.R_range)} R values")

        for Q in self.Q_range:
            for R in self.R_range:
                score = self._evaluate_params(y_train, x_train, Q, R)
                results.append({'Q': Q, 'R': R, 'score': score})

                if verbose:
                    print(f"  Q={Q:.1e}, R={R:.2f} → score={score:.4f}")

        # Find best parameters (minimum score)
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmin()
        best_params = results_df.loc[best_idx]

        if verbose:
            print(f"\n✓ Best parameters: Q={best_params['Q']:.1e}, R={best_params['R']:.2f}")

        return best_params['Q'], best_params['R'], results_df