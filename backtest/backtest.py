import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import yfinance as yf
import logging
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    ticker: str
    buy_date: datetime
    sell_date: datetime
    allocation_percentage: float
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    buy_amount: Optional[float] = None
    sell_amount: Optional[float] = None
    profit_loss: Optional[float] = None
    quantity: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'Ticker': self.ticker,
            'Buy Date': self.buy_date.strftime('%Y-%m-%d'),
            'Sell Date': self.sell_date.strftime('%Y-%m-%d'),
            'Allocation %': f"{self.allocation_percentage}%",
            'Buy Price': f"${self.buy_price:.2f}" if self.buy_price else 'N/A',
            'Sell Price': f"${self.sell_price:.2f}" if self.sell_price else 'N/A',
            'Quantity': f"{self.quantity:.4f}" if self.quantity else 'N/A',
            'Buy Amount': f"${self.buy_amount:.2f}" if self.buy_amount else 'N/A',
            'Sell Amount': f"${self.sell_amount:.2f}" if self.sell_amount else 'N/A',
            'Profit/Loss': f"${self.profit_loss:.2f}" if self.profit_loss else 'N/A',
            'Return %': f"{((self.sell_price/self.buy_price - 1) * 100):.2f}%" if (self.sell_price and self.buy_price) else 'N/A'
        }

class StockAccount:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.holdings: Dict[str, float] = {}  # ticker -> quantity
        self.trades_executed = []
        self.trades_skipped = []

    def can_execute_buy(self, trade: Trade, price: float) -> bool:
        required_amount = trade.buy_amount
        return self.cash >= required_amount

    def execute_buy(self, trade: Trade, price: float):
        if not self.can_execute_buy(trade, price):
            self.trades_skipped.append(trade)
            return False

        self.cash -= trade.buy_amount
        logger.info(f"Cash after buy: {self.cash}")
        self.holdings[trade.ticker] = self.holdings.get(trade.ticker, 0.0) + trade.quantity
        self.trades_executed.append(trade)
        return True

    def execute_sell(self, trade: Trade, price: float):
        if trade.ticker not in self.holdings or self.holdings[trade.ticker] < trade.quantity:
            return False

        self.cash += trade.sell_amount
        logger.info(f"Cash after sell: {self.cash}")
        self.holdings[trade.ticker] -= trade.quantity
        if abs(self.holdings[trade.ticker]) < 0.0001:  # Using small epsilon for float comparison
            del self.holdings[trade.ticker]
        return True

class Backtest:
    def __init__(self, initial_cash: float = 10000.0):
        self.account = StockAccount(initial_cash)
        self.trades: list[Trade] = []
        self.price_cache = {}  # Cache for price data

    def load_trades_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            trade = Trade(
                ticker=row['ticker'],
                buy_date=pd.to_datetime(row['buy_date']),
                sell_date=pd.to_datetime(row['sell_date']),
                allocation_percentage=row['allocation_percentage']
            )
            self.trades.append(trade)
        logger.info(f"Loaded {len(self.trades)} trades from CSV")

    def get_stock_price(self, ticker: str, date: datetime, trade: Trade) -> float:
        try:
            # Check if we have cached data for this ticker and trade period
            cache_key = f"{ticker}_{trade.buy_date.date()}_{trade.sell_date.date()}"
            
            if cache_key not in self.price_cache:
                # Get historical data for the entire trade period
                stock = yf.Ticker(ticker)
                # Add a small buffer to ensure we have the dates we need
                start_date = trade.buy_date - timedelta(days=1)
                end_date = trade.sell_date + timedelta(days=1)
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty:
                    raise ValueError(f"No price data found for {ticker} between {start_date} and {end_date}")
                
                # Convert timezone-aware index to timezone-naive
                hist.index = hist.index.tz_localize(None)
                
                self.price_cache[cache_key] = hist
                logger.info(f"Fetched price data for {ticker} from {start_date} to {end_date}")

            # Get the closest date from our cached data
            hist = self.price_cache[cache_key]
            closest_date = min(hist.index, key=lambda x: abs(x - date))
            return hist.loc[closest_date, 'Close']
            
        except Exception as e:
            logger.error(f"Error getting price for {ticker} on {date}: {str(e)}")
            raise

    def execute_trades(self):
        for i, trade in enumerate(self.trades, 1):
            try:
                logger.info(f"Processing trade {i}/{len(self.trades)}: {trade.ticker}")
                
                # Get buy price and calculate buy amount
                buy_price = self.get_stock_price(trade.ticker, trade.buy_date, trade)
                trade.buy_price = buy_price
                logger.info(f"Buy price: {buy_price}")
                logger.info(f"Cash before buy: {self.account.cash}")
                logger.info(f"Allocation percentage: {trade.allocation_percentage}")
                trade.buy_amount = (self.account.cash * trade.allocation_percentage / 100)
                trade.quantity = trade.buy_amount / buy_price
                trade.buy_amount = trade.quantity * buy_price  # Adjust for actual quantity
                logger.info(f"Buy amount: {trade.buy_amount} for {trade.quantity:.4f} shares")

                # Execute buy
                if not self.account.execute_buy(trade, buy_price):
                    logger.warning(f"Skipping trade for {trade.ticker} - insufficient funds")
                    continue

                # Get sell price and calculate sell amount
                sell_price = self.get_stock_price(trade.ticker, trade.sell_date, trade)
                trade.sell_price = sell_price
                trade.sell_amount = trade.quantity * sell_price
                logger.info(f"Sell amount: {trade.sell_amount}")
                trade.profit_loss = trade.sell_amount - trade.buy_amount
                logger.info(f"Profit/Loss: {trade.profit_loss}")
                # Execute sell
                if not self.account.execute_sell(trade, sell_price):
                    logger.warning(f"Failed to execute sell for {trade.ticker}")
                    continue

                logger.info(f"Successfully executed trade for {trade.ticker}: "
                          f"Buy: ${trade.buy_price:.2f}, Sell: ${trade.sell_price:.2f}, "
                          f"P/L: ${trade.profit_loss:.2f}")

            except Exception as e:
                logger.error(f"Error processing trade for {trade.ticker}: {str(e)}")
                self.account.trades_skipped.append(trade)

    def get_results(self):
        total_profit_loss = sum(trade.profit_loss for trade in self.account.trades_executed if trade.profit_loss is not None)
        return {
            'final_cash': self.account.cash,
            'current_holdings': self.account.holdings,
            'trades_executed': len(self.account.trades_executed),
            'trades_skipped': len(self.account.trades_skipped),
            'total_profit_loss': total_profit_loss,
            'trades': self.account.trades_executed
        }

    def print_trade_summary(self):
        # Convert trades to DataFrame
        trades_data = [trade.to_dict() for trade in self.account.trades_executed]
        if trades_data:
            df = pd.DataFrame(trades_data)
            print("\nTrade Summary:")
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Print skipped trades
        if self.account.trades_skipped:
            skipped_tickers = [trade.ticker for trade in self.account.trades_skipped]
            print(f"\nSkipped Trades: {', '.join(skipped_tickers)}")

        # Print account summary
        print("\nAccount Summary:")
        summary_data = [
            ["Initial Cash", f"${self.account.initial_cash:,.2f}"],
            ["Final Cash", f"${self.account.cash:,.2f}"],
            ["Total Profit/Loss", f"${self.account.cash - self.account.initial_cash:,.2f}"],
            ["Return %", f"{((self.account.cash/self.account.initial_cash - 1) * 100):.2f}%"],
            ["Trades Executed", len(self.account.trades_executed)],
            ["Trades Skipped", len(self.account.trades_skipped)]
        ]
        print(tabulate(summary_data, tablefmt='grid'))

        # Print current holdings
        if self.account.holdings:
            print("\nCurrent Holdings:")
            holdings_data = [[ticker, f"{quantity:.4f}"] for ticker, quantity in self.account.holdings.items()]
            print(tabulate(holdings_data, headers=['Ticker', 'Quantity'], tablefmt='grid'))

def main():
    # Example usage
    backtest = Backtest(initial_cash=10000.0)
    backtest.load_trades_from_csv('example/stock_signals.csv')
    backtest.execute_trades()
    backtest.print_trade_summary()

if __name__ == "__main__":
    main()
