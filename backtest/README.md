# Trading Strategy Backtester

A Python-based backtesting tool for evaluating trading strategies with support for dip-buying strategies.

## Features

- Backtest multiple trades with different allocation percentages
- Automatic dip-buying strategy implementation
- Technical indicator support (SMA20)
- Detailed performance metrics and reporting
- Support for custom initial balance
- CSV-based trade input

## Requirements

- Python 3.8+
- pandas
- yfinance

## Installation

The backtester uses a virtual environment managed by the project's Makefile. To set up:

```bash
make setup
```

## Usage

### Input CSV Format

Create a CSV file with the following columns:
- `ticker`: Stock symbol (e.g., "AAPL")
- `sell_date`: Target sell date (YYYY-MM-DD)
- `allocation_percentage`: Percentage of available balance to invest
- `buy_date` (optional): Specific buy date (YYYY-MM-DD). If not provided, uses current date.

Example CSV:
```csv
ticker,sell_date,allocation_percentage,buy_date
AAPL,2024-12-31,50,2024-01-01
MSFT,2024-12-31,50,2024-01-01
```

### Running the Backtester

You can run the backtester like below

1. Running the script directly:
```bash
./run_backtest.sh --input trades.csv --initial_balance 10000 --detailed
```

Arguments:
- `--input`: Path to input CSV file (required)
- `--initial_balance`: Starting account balance (default: 10000)
- `--detailed`: Show detailed trade results including dip buys

### Dip-Buying Strategy

The backtester implements an automatic dip-buying strategy with the following rules:
1. Price is below 20-day SMA (downtrend)
2. Price has dropped at least 10% from recent high
3. 20-day SMA is still above initial buy price (overall uptrend)
4. Maximum of 2 additional buys per position
5. Only buys if price is below initial buy price
6. Minimum 5 trading days between dip buys

## Output

The backtester provides detailed performance metrics including:
- Total return and percentage
- Trade statistics (win rate, average profit/loss)
- Time period analysis
- Individual trade results (with `--detailed` flag)
- Dip-buying statistics

