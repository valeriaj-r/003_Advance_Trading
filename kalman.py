"""
Kalman Filter module: Sequential Decision Process framework
Following Powell's framework with 5 core elements + 6-step modeling process
"""
import numpy as np
import pandas as pd


class KalmanFilterSDP:
    """
    Kalman Filter as Sequential Decision Process

    POWELL'S 5 ELEMENTS:
    1. State Variables (St):
       - Rt (Physical): hedge_ratio (beta)
       - It (Information): spread, spread_volatility
       - Bt (Belief): covariance P (uncertainty about hedge ratio)

    2. Decision Variables (xt):
       - Trading decision: {LONG_A_SHORT_B, SHORT_A_LONG_B, NO_POSITION}

    3. Exogenous Information (Wt+1):
       - New price observations: price_A(t+1), price_B(t+1)

    4. Transition Function SM(St, xt, Wt+1):
       - Kalman prediction: beta(t+1|t) = beta(t|t)
       - Kalman update: beta(t+1|t+1) based on new observations
       - Spread update: spread(t+1) = log(A) - beta * log(B)

    5. Objective Function:
       - Maximize E[Σ Ct(St, xt)] where Ct = PnL - costs
    """

    def __init__(self, Q=1e-4, R=1e-2):
        """
        Initialize Kalman Filter

        Parameters:
        -----------
        Q : float
            Process noise covariance (how much beta can change)
        R : float
            Measurement noise covariance (trust in observations)
        """
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise

        # State variables
        self.beta = None  # Hedge ratio (Rt - Physical State)
        self.P = None  # Covariance (Bt - Belief State)

        # History
        self.beta_history = []
        self.spread_history = []
        self.P_history = []

    def initialize(self, initial_beta, initial_P=1.0):
        """Initialize state S0"""
        self.beta = initial_beta
        self.P = initial_P
        self.beta_history = [initial_beta]
        self.P_history = [initial_P]

    def predict(self):
        """
        Prediction step: SM(St, xt, Wt+1) - Time update
        State transition: beta(t+1|t) = beta(t|t)
        Covariance: P(t+1|t) = P(t|t) + Q
        """
        # No change in beta (random walk)
        self.P = self.P + self.Q

    def update(self, log_price_A, log_price_B):
        """
        Update step: Incorporate exogenous information Wt+1

        Parameters:
        -----------
        log_price_A : float
            New observation of log price A
        log_price_B : float
            New observation of log price B
        """
        # Innovation: difference between observed and predicted
        y = log_price_A - self.beta * log_price_B

        # Innovation covariance
        S = log_price_B ** 2 * self.P + self.R

        # Kalman gain
        K = self.P * log_price_B / S

        # Update state
        self.beta = self.beta + K * y

        # Update covariance
        self.P = (1 - K * log_price_B) * self.P

        # Store history
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)

        return self.beta

    def compute_spread(self, log_price_A, log_price_B):
        """
        Compute spread (Information State It)
        spread = log(A) - beta * log(B)
        """
        spread = log_price_A - self.beta * log_price_B
        self.spread_history.append(spread)
        return spread

    def compute_zscore(self, window=30):
        """
        Compute z-score for decision making (Policy component)

        Returns z-score of current spread relative to rolling window
        """
        if len(self.spread_history) < window:
            return 0.0

        recent_spreads = np.array(self.spread_history[-window:])
        mean = np.mean(recent_spreads)
        std = np.std(recent_spreads)

        if std < 1e-6:
            return 0.0

        current_spread = self.spread_history[-1]
        zscore = (current_spread - mean) / std

        return zscore

    def get_state(self):
        """
        Get current state St = {Rt, It, Bt}

        Returns:
        --------
        dict with:
        - beta (Rt): hedge ratio
        - spread (It): current spread
        - P (Bt): uncertainty
        """
        return {
            "beta": self.beta,
            "spread": self.spread_history[-1] if self.spread_history else 0,
            "P": self.P
        }


def create_policy(entry_threshold, exit_threshold):
    """
    Policy function: X^π(St) -> decision

    MEAN REVERSION LOGIC:
    - When z-score is HIGH (+): Spread is above mean → SHORT spread (expect decrease)
    - When z-score is LOW (-): Spread is below mean → LONG spread (expect increase)

    Decision based on z-score:
    - z > entry_threshold: SHORT spread (SHORT A, LONG B) - spread too HIGH
    - z < -entry_threshold: LONG spread (LONG A, SHORT B) - spread too LOW
    - |z| < exit_threshold: EXIT position - spread returned to mean
    - otherwise: HOLD current position

    Parameters:
    -----------
    entry_threshold : float
        Z-score threshold to enter position (e.g., 2.0)
    exit_threshold : float
        Z-score threshold to exit position (e.g., 0.5)

    Returns:
    --------
    function that takes (zscore, current_position) and returns decision
    """

    def policy(zscore, current_position):
        """
        Decision function

        Returns:
        --------
        int: 1 (LONG spread), -1 (SHORT spread), 0 (NO position)
        """
        # Exit logic - only exit if we have a position AND spread reverted
        if current_position != 0:
            # Exit when spread returns close to mean
            if abs(zscore) < exit_threshold:
                return 0  # Exit position

            # Also exit if spread moved too far against us (stop-loss logic)
            if current_position == 1 and zscore > entry_threshold * 1.5:
                return 0  # Exit long spread if it went even more negative
            if current_position == -1 and zscore < -entry_threshold * 1.5:
                return 0  # Exit short spread if it went even more positive

        # Entry logic - only enter if no position
        if current_position == 0:
            if zscore > entry_threshold:
                return -1  # SHORT spread (spread is HIGH, expect mean reversion DOWN)
            elif zscore < -entry_threshold:
                return 1  # LONG spread (spread is LOW, expect mean reversion UP)

        # Hold current position if no exit/entry signals
        return current_position

    return policy


if __name__ == "__main__":
    # Test Kalman Filter
    np.random.seed(42)

    # Simulate data
    n = 100
    true_beta = 1.5
    price_B = np.cumsum(np.random.randn(n)) + 100
    price_A = true_beta * price_B + np.random.randn(n) * 2

    # Initialize Kalman
    kf = KalmanFilterSDP(Q=1e-4, R=1e-2)
    kf.initialize(initial_beta=1.0)

    # Run filter
    for i in range(1, n):
        kf.predict()
        kf.update(np.log(price_A[i]), np.log(price_B[i]))
        spread = kf.compute_spread(np.log(price_A[i]), np.log(price_B[i]))

    print(f"True beta: {true_beta:.4f}")
    print(f"Estimated beta: {kf.beta:.4f}")
    print(f"Final uncertainty (P): {kf.P:.6f}")