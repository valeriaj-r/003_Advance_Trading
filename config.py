"""
Configuration file for Pairs Trading Strategy
"""

# Data parameters
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

# Data splits (chronological)
TRAIN_RATIO = 0.60
TEST_RATIO = 0.20
VALIDATION_RATIO = 0.20

# Trading parameters
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE_PCT = 0.80  # Use 80% of available capital

# Transaction costs
COMMISSION_RATE = 0.00125  # 0.125% per trade
BORROW_RATE_ANNUAL = 0.0025  # 0.25% annual for short positions
BORROW_RATE_DAILY = BORROW_RATE_ANNUAL / 252  # Daily rate

# Cointegration parameters
COINTEGRATION_ALPHA = 0.05
USE_LOG_PRICES = True

# Kalman Filter grid search parameters
Q_VALUES = [1e-5, 5e-5, 1e-4]  # Process noise (smaller = more stable beta)
R_VALUES = [1e-2, 5e-2, 1e-1]  # Measurement noise (larger = trust observations less)

# Z-score threshold grid search parameters
ENTRY_THRESHOLDS = [2.0, 2.5, 3.0]  # More conservative entry (higher threshold)
EXIT_THRESHOLDS = [0.5, 0.75, 1.0]  # Exit closer to mean

# Asset universe by sector
SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "AVGO", "ORCL", "INTC", "AMD", "CRM"],
    "Financials": ["JPM", "V", "MA", "C", "BAC", "BRK-B", "WFC", "PYPL"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "TMO", "DHR"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "Consumer Staples": ["WMT", "KO", "PEP", "PG", "COST", "MO"],
    "Communication Services": ["META", "NFLX", "DIS", "VZ", "T", "CMCSA"],
    "Energy": ["XOM", "CVX", "SLB"],
    "Industrials": ["HON", "DE"]
}