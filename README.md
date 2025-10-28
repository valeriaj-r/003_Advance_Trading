# Pairs Trading Strategy with Kalman Filter

Statistical arbitrage strategy using pairs trading with dynamic hedge ratio estimation via Kalman Filter, formulated as a Sequential Decision Process following Powell's framework.

## Project Overview

This project implements a market-neutral pairs trading strategy that:
- Identifies cointegrated asset pairs using statistical tests
- Estimates dynamic hedge ratios using Kalman filters
- Generates trading signals based on mean reversion
- Implements realistic transaction costs and borrowing fees

## Sequential Decision Process Framework (Powell)

### 6-Step Modeling Process

1. *Narrative*: Statistical arbitrage exploiting temporary deviations from equilibrium
2. *Core Elements*: Returns, trading decisions, price uncertainty, parameter uncertainty
3. *Mathematical Model*: State variables, decisions, exogenous info, transitions, objective
4. *Uncertainty Model*: Initial beta estimate, daily price observations
5. *Policy Design*: Grid search over Kalman (Q,R) and threshold parameters
6. *Policy Evaluation*: Backtesting on train/test/validation splits

### 5 Core Elements

- *State Variables (St)*: Hedge ratio, spread, covariance matrix
- *Decision Variables (xt)*: {Long spread, Short spread, No position}
- *Exogenous Information (Wt+1)*: New price observations
- *Transition Function*: Kalman prediction and update equations
- *Objective Function*: Maximize E[Σ PnL - transaction costs - borrow costs]

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- numpy
- pandas
- statsmodels
- matplotlib
- yfinance

## Project Structure

```
pairs_trading/
├── requirements.txt       # Dependencies
├── README.md             # This file
├── config.py             # Configuration parameters
├── data.py               # Data download and splitting
├── cointegration.py      # Cointegration testing
├── kalman.py             # Kalman Filter as SDP
├── backtesting.py        # Strategy execution with costs
├── visualizations.py     # Performance visualizations
└── main.py               # Main orchestrator
```

## Usage

Run the complete strategy:

bash
python main.py


This will:
1. Download 15 years of historical data (2010-2025)
2. Test all pairs within sectors for cointegration
3. Select best cointegrated pair
4. Optimize parameters on training data (60%)
5. Evaluate on test data (20%)
6. Validate on validation data (20%)
7. Display visualizations and performance metrics

## Configuration

Edit config.py to modify:
- Asset universe and sectors
- Date range
- Initial capital ($1,000,000 default)
- Transaction costs (0.125% per trade)
- Borrow rate (0.25% annual)
- Grid search ranges for Q, R, and thresholds

## Strategy Parameters

### Cointegration Testing
- ADF test with α = 0.05
- Tests performed within sectors only
- Requires both series non-stationary and residuals stationary

### Kalman Filter
- *Q (Process Noise)*: [1e-5, 1e-4, 1e-3]
- *R (Measurement Noise)*: [1e-3, 1e-2, 1e-1]

### Trading Thresholds
- *Entry Threshold*: [1.5, 2.0, 2.5, 3.0] standard deviations
- *Exit Threshold*: [0.3, 0.5, 0.7, 1.0] standard deviations

## Outputs

### Visualizations
1. *Z-Score with Trading Signals* 
   - Red dashed lines: Entry thresholds (±entry_threshold)
   - Green dashed lines: Exit thresholds (±exit_threshold)
   - Green triangles (▲): LONG spread entries (z-score was LOW, expecting increase)
   - Red triangles (▼): SHORT spread entries (z-score was HIGH, expecting decrease)
   - Orange X: Exit signals (spread returned to mean)
   - Shaded regions: Trading zones
2. *Spread Evolution*: Spread over time with position overlays
3. *Portfolio Value*: Capital evolution over time


### Performance Metrics
- Total Return (%)
- Sharpe Ratio (annualized)
- Maximum Drawdown (%)
- Number of Trades
- Win Rate (%)
- Mean/Std Daily Returns

### Output for Each Period
Results are displayed for:
- *Train*: Parameter optimization (60% of data)
- *Test*: Out-of-sample evaluation (20% of data)
- *Validation*: Final validation (20% of data)

## Transaction Costs

- *Commission*: 0.125% per trade (both entry and exit)
- *Borrow Cost*: 0.25% annual rate on short positions
- *Daily Accrual*: Borrow costs calculated daily
- *Position Sizing*: 80% of available capital

## Asset Universe

### Sectors Included
- Technology (10 assets)
- Financials (8 assets)
- Healthcare (8 assets)
- Consumer Discretionary (7 assets)
- Consumer Staples (6 assets)
- Communication Services (6 assets)
- Energy (3 assets)
- Industrials (2 assets)

## Notes

- Data automatically downloaded from Yahoo Finance
- No retraining during test/validation periods
- Walk-forward analysis not implemented (simple holdout)
- All pairs tested within same sector only
- Uses log prices for cointegration testing and spread calculation

## Authors

Mónica Valeria Jáuregui Rodríguez
Jimena Argüelles Perez


## License

Educational project - use at your own risk


