# Artemis Trading System

Artemis is a sophisticated trading system that combines automated trading with comprehensive backtesting capabilities. The system is designed to execute trades based on predefined signals while incorporating advanced features like dip-buying strategies and technical analysis.

## Components

### 1. Backtesting System

The backtesting system allows you to test trading strategies using historical data. It provides detailed analysis of trade performance and account metrics.

#### Key Features:
- **Signal-based Trading**: Execute trades based on predefined buy/sell signals
- **Dip-Buying Strategy**: Implements an intelligent dip-buying mechanism with the following conditions:
  - Price below 20-day SMA (downtrend)
  - Price dropped at least 10% from recent high
  - 20-day SMA above current price (overall uptrend)
  - Current price below initial buy price
  - Minimum 5 days between buys
- **Position Management**: 
  - Maximum of 3 buys per position (initial + 2 dip buys)
  - Same allocation percentage for each buy
  - Fractional share support
- **Performance Metrics**:
  - Total return
  - Annualized return
  - Individual trade performance
  - Position tracking
  - Cash management

#### Usage:
```bash
./backtest/run_backtest.sh --input example/stock_signals.csv  --initial_balance 10000
```

#### Input Format:
Create a CSV file with the following columns:
```
ticker,buy_date,sell_date,allocation_percentage
AAPL,2023-01-01,2023-12-31,10
MSFT,2023-01-01,2023-12-31,10
```

#### Output:
The backtest generates a detailed report including:
- Trade summary with individual trade details
- Account performance metrics
- Position tracking
- Cash management statistics

### 2. Trading Bot (AWS Lambda)

The trading bot runs on AWS Lambda and executes real-time trades through Alpaca Markets API. It's designed to be serverless, scalable, and cost-effective.

#### Prerequisites:
1. **Alpaca Markets Account**:
   - Sign up for an Alpaca Markets account at https://alpaca.markets/
   - Get your API keys (both paper trading and live trading)
   - Enable Alpaca Markets API access

2. **AWS Account**:
   - Create an AWS account if you don't have one
   - Set up AWS CLI and configure credentials

#### AWS Setup:

1. **IAM Role Creation**:
   - Create an IAM role for Lambda execution
   - Attach necessary policies for DynamoDB, CloudWatch, and EventBridge access

2. **DynamoDB Setup**:
   - Create a table named `trading_strategy`
   - Use `ticker` as partition key and `buy_date` as sort key
   - Enable on-demand capacity mode

3. **Lambda Function**:
   - Create a new Lambda function named `ArtemisTradingBot`
   - Use Go runtime/AL2023 base image
   - Set memory to 256MB and timeout to 5 minutes
   - Configure environment variables:
     ```
     ALPACA_API_KEY=your_api_key
     ALPACA_SECRET_KEY=your_secret_key
     ```
   - Deploy using `make deploy`

4. **EventBridge Rules**:
   - Create a rule to trigger Lambda at market open (12 noon ET)
   - Set the target as the Lambda function

#### Environment Setup:

1. **Local Development**:
   Create a `.env` file:
   ```
   ALPACA_API_KEY=your_api_key
   ALPACA_SECRET_KEY=your_secret_key
   ```

2. **AWS Lambda**:
   - Set environment variables in Lambda configuration
   - Use AWS Secrets Manager for production deployments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
