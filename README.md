# 003_Advance_Trading

## Valeria Jauregui & Jimena Argüelles

# 📈 Pairs Trading Strategy with Kalman Filter

Statistical arbitrage strategy using pairs trading with dynamic hedge ratio estimation via Kalman filters, formulated as a sequential decision process.

---

## 📋 Overview

This project implements a market-neutral pairs trading strategy that:

- Identifies cointegrated asset pairs using Engle-Granger methodology  
- Estimates dynamic hedge ratios using Kalman filters (formulated as sequential decision processes)  
- Generates mean-reversion trading signals based on Z-scores  
- Backtests with realistic transaction costs and borrowing fees  

---

## 🏗️ Project Structure

```plaintext
pairs_trading/
├── data_handler.py       # Data loading and train/test/val splits
├── cointegration.py      # Cointegration testing (Engle-Granger)
├── kalman_filter.py      # Kalman filter as sequential decision process
├── strategy.py           # Trading strategy and signal generation
├── backtester.py         # Backtesting engine with realistic costs
├── visualizer.py         # Plotting functions
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
---
## 🔧 Installation

### Prerequisites

- Python 3.8 or higher  
- pip package manager  

### Setup

1. Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd pairs_trading
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

### Execution

Simply run:

```bash
python main.py
```

The script will:

1. Download data from Yahoo Finance  
2. Find cointegrated pairs  
3. Optimize Kalman parameters  
4. Backtest the strategy  
5. Generate all plots  
6. Print performance metrics  


