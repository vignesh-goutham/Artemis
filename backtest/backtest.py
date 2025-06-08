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
    buy_count: int = 0  # Track number of buys for this position
    last_buy_date: Optional[datetime] = None  # Track last buy date for this position
    hist_data: Optional[pd.DataFrame] = None  # Store historical data for technical analysis
    initial_buy_price: Optional[float] = None  # Track the initial buy price for dip calculations

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
            'Return %': f"{((self.sell_price/self.buy_price - 1) * 100):.2f}%" if (self.sell_price and self.buy_price and self.buy_price != 0) else 'N/A',
            'Buy Count': self.buy_count
        }

@dataclass
class Signal:
    ticker: str
    initial_buy_date: datetime
    sell_date: datetime
    allocation_percentage: float
    trades: List[Trade] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    def add_trade(self, trade: Trade):
        self.trades.append(trade)

    def get_total_quantity(self) -> float:
        return sum(trade.quantity for trade in self.trades if trade.quantity is not None)

    def get_total_buy_amount(self) -> float:
        return sum(trade.buy_amount for trade in self.trades if trade.buy_amount is not None)

    def get_weighted_avg_buy_price(self) -> float:
        total_quantity = self.get_total_quantity()
        if total_quantity == 0:
            return 0
        return self.get_total_buy_amount() / total_quantity

class StockAccount:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.holdings: Dict[str, float] = {}  # ticker -> quantity
        self.signals: List[Signal] = []  # List of signals
        self.trades_skipped = []

    def can_execute_buy(self, trade: Trade, price: float, hist_data: pd.DataFrame) -> bool:
        required_amount = trade.buy_amount
        
        # Check if we've already made 3 buys for this position
        if trade.buy_count >= 3:
            logger.info(f"Skipping buy for {trade.ticker} - maximum buys (3) reached")
            return False

        # Check if we have enough cash
        if self.cash < required_amount:
            logger.info(f"Skipping buy for {trade.ticker} - insufficient funds")
            return False

        # For first buy, just check cash
        if trade.buy_count == 0:
            return True

        # For additional buys, check dip buying conditions
        current_price = price
        initial_price = trade.initial_buy_price  # Use the initial buy price for dip calculations
        last_buy_date = trade.last_buy_date

        # Calculate technical indicators
        sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        recent_high = hist_data['High'].rolling(window=20).max().iloc[-1]
        price_drop = (recent_high - current_price) / recent_high * 100

        # Check dip buying conditions
        conditions_met = [
            current_price < sma_20,  # Price below 20-day SMA (downtrend)
            price_drop >= 10,  # Price dropped at least 10% from recent high
            sma_20 > current_price,  # 20-day SMA above current price (overall uptrend)
            current_price < initial_price,  # Current price below initial buy price
        ]

        # Check if enough days have passed since last buy
        if last_buy_date:
            days_since_last_buy = (trade.buy_date - last_buy_date).days
            conditions_met.append(days_since_last_buy >= 5)

        # Debug logging for dip buy conditions
        logger.info(f"\nDip buy conditions for {trade.ticker} on {trade.buy_date.date()}:")
        logger.info(f"Current price: ${current_price:.2f}")
        logger.info(f"Initial price: ${initial_price:.2f}")
        logger.info(f"20-day SMA: ${sma_20:.2f}")
        logger.info(f"Recent high: ${recent_high:.2f}")
        logger.info(f"Price drop: {price_drop:.2f}%")
        logger.info(f"Days since last buy: {(trade.buy_date - last_buy_date).days if last_buy_date else 'N/A'}")
        logger.info(f"Conditions met: {conditions_met}")

        return all(conditions_met)

    def execute_buy(self, trade: Trade, price: float) -> bool:
        if not self.can_execute_buy(trade, price, trade.hist_data):
            return False

        self.cash -= trade.buy_amount
        self.holdings[trade.ticker] = self.holdings.get(trade.ticker, 0.0) + trade.quantity
        trade.buy_count += 1
        trade.last_buy_date = trade.buy_date
        return True

    def execute_sell(self, signal: Signal, sell_price: float) -> bool:
        total_quantity = signal.get_total_quantity()
        if signal.ticker not in self.holdings or self.holdings[signal.ticker] < total_quantity:
            return False

        # Calculate sell amount and update each trade
        sell_amount = total_quantity * sell_price
        self.cash += sell_amount

        # Update each trade in the signal
        for trade in signal.trades:
            trade.sell_price = sell_price
            trade.sell_amount = trade.quantity * sell_price
            trade.profit_loss = trade.sell_amount - trade.buy_amount

        # Clear the position
        self.holdings[signal.ticker] -= total_quantity
        if abs(self.holdings[signal.ticker]) < 0.0001:  # Using small epsilon for float comparison
            del self.holdings[signal.ticker]

        return True

class Backtest:
    def __init__(self, initial_cash: float = 10000.0):
        self.account = StockAccount(initial_cash)
        self.signals: List[Signal] = []
        self.price_cache = {}  # Cache for price data

    def load_trades_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            signal = Signal(
                ticker=row['ticker'],
                initial_buy_date=pd.to_datetime(row['buy_date']),
                sell_date=pd.to_datetime(row['sell_date']),
                allocation_percentage=row['allocation_percentage']
            )
            self.signals.append(signal)
        logger.info(f"Loaded {len(self.signals)} signals from CSV")

    def get_stock_price(self, ticker: str, date: datetime, signal: Signal) -> float:
        try:
            # Check if we have cached data for this ticker and signal period
            cache_key = f"{ticker}_{signal.initial_buy_date.date()}_{signal.sell_date.date()}"
            
            if cache_key not in self.price_cache:
                # Get historical data for the entire signal period
                stock = yf.Ticker(ticker)
                # Add a small buffer to ensure we have the dates we need
                start_date = signal.initial_buy_date - timedelta(days=30)  # Get more data for SMA calculation
                end_date = signal.sell_date + timedelta(days=1)
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
        for i, signal in enumerate(self.signals, 1):
            try:
                logger.info(f"\nProcessing signal {i}/{len(self.signals)}: {signal.ticker}")
                
                # Create and execute initial buy
                initial_trade = Trade(
                    ticker=signal.ticker,
                    buy_date=signal.initial_buy_date,
                    sell_date=signal.sell_date,
                    allocation_percentage=signal.allocation_percentage,
                    buy_count=0
                )
                
                # Get initial buy price and execute first buy
                buy_price = self.get_stock_price(signal.ticker, signal.initial_buy_date, signal)
                initial_trade.buy_price = buy_price
                initial_trade.initial_buy_price = buy_price  # Set initial buy price for dip calculations
                initial_trade.hist_data = self.price_cache[f"{signal.ticker}_{signal.initial_buy_date.date()}_{signal.sell_date.date()}"]
                initial_trade.buy_amount = (self.account.cash * signal.allocation_percentage / 100)
                initial_trade.quantity = initial_trade.buy_amount / buy_price
                initial_trade.buy_amount = initial_trade.quantity * buy_price  # Adjust for actual quantity

                # Execute initial buy
                if not self.account.execute_buy(initial_trade, buy_price):
                    logger.warning(f"Skipping initial buy for {signal.ticker} - insufficient funds")
                    self.account.trades_skipped.append(initial_trade)
                    continue

                signal.add_trade(initial_trade)

                # Check for dip-buying opportunities each day until sell date
                current_date = signal.initial_buy_date + timedelta(days=1)
                while current_date < signal.sell_date:
                    try:
                        # Get price for current date
                        current_price = self.get_stock_price(signal.ticker, current_date, signal)
                        
                        # Create a new trade object for the dip buy
                        dip_trade = Trade(
                            ticker=signal.ticker,
                            buy_date=current_date,
                            sell_date=signal.sell_date,
                            allocation_percentage=signal.allocation_percentage,
                            buy_count=len(signal.trades),
                            last_buy_date=signal.trades[-1].buy_date if signal.trades else None,
                            hist_data=initial_trade.hist_data,
                            buy_price=current_price,  # Set current price as the buy price for this dip trade
                            initial_buy_price=initial_trade.buy_price  # Set initial buy price for dip calculations
                        )
                        
                        # Calculate buy amount
                        dip_trade.buy_amount = (self.account.cash * signal.allocation_percentage / 100)
                        dip_trade.quantity = dip_trade.buy_amount / current_price
                        dip_trade.buy_amount = dip_trade.quantity * current_price
                        
                        # Try to execute dip buy
                        if self.account.execute_buy(dip_trade, current_price):
                            signal.add_trade(dip_trade)
                            logger.info(f"Executed dip buy for {signal.ticker} on {current_date.date()} at ${current_price:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Error checking dip buy for {signal.ticker} on {current_date}: {str(e)}")
                        self.account.trades_skipped.append(dip_trade)
                    
                    current_date += timedelta(days=1)

                # Get sell price and execute sell for all trades in the signal
                sell_price = self.get_stock_price(signal.ticker, signal.sell_date, signal)
                
                # Execute sell for all trades in the signal
                if not self.account.execute_sell(signal, sell_price):
                    logger.warning(f"Failed to execute sell for {signal.ticker}")
                    self.account.trades_skipped.append(signal.trades[0])  # Add first trade to skipped
                    continue

                logger.info(f"Successfully executed signal for {signal.ticker}: "
                          f"Initial Buy: ${initial_trade.buy_price:.2f}, "
                          f"Sell: ${sell_price:.2f}, "
                          f"Total P/L: ${sum(t.profit_loss for t in signal.trades):.2f}")

            except Exception as e:
                logger.error(f"Error processing signal for {signal.ticker}: {str(e)}")
                self.account.trades_skipped.append(signal.trades[0] if signal.trades else None)

    def print_trade_summary(self):
        # Convert all trades to DataFrame
        trades_data = []
        for signal in self.signals:
            for trade in signal.trades:
                if trade.buy_price is not None and trade.sell_price is not None:
                    trades_data.append(trade.to_dict())
                
        if trades_data:
            df = pd.DataFrame(trades_data)
            print("\nTrade Summary:")
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Print skipped trades
        if self.account.trades_skipped:
            skipped_tickers = [trade.ticker for trade in self.account.trades_skipped if trade is not None]
            print(f"\nSkipped Trades: {', '.join(skipped_tickers)}")

        # Calculate trading period
        if self.signals:
            start_date = min(signal.initial_buy_date for signal in self.signals)
            end_date = max(signal.sell_date for signal in self.signals)
            trading_days = (end_date - start_date).days
            years = trading_days / 365.25  # Using 365.25 to account for leap years
            
            # Calculate annualized return
            total_return = (self.account.cash / self.account.initial_cash) - 1
            annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        else:
            years = 0
            annualized_return = 0

        # Print account summary
        print("\nAccount Summary:")
        summary_data = [
            ["Initial Cash", f"${self.account.initial_cash:,.2f}"],
            ["Final Cash", f"${self.account.cash:,.2f}"],
            ["Total Profit/Loss", f"${self.account.cash - self.account.initial_cash:,.2f}"],
            ["Total Return %", f"{((self.account.cash/self.account.initial_cash - 1) * 100):.2f}%"],
            ["Annualized Return %", f"{(annualized_return * 100):.2f}%"],
            ["Trading Period", f"{years:.2f} years"],
            ["Trades Executed", sum(len(signal.trades) for signal in self.signals)],
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
