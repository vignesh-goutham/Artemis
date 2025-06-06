import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

class Trade:
    def __init__(self, ticker: str, buy_date: str, sell_date: str, allocation_percentage: float):
        self.ticker = ticker
        # Convert dates to timezone-naive datetime objects
        self.buy_date = pd.to_datetime(buy_date).tz_localize(None)
        self.sell_date = pd.to_datetime(sell_date).tz_localize(None)
        self.allocation_percentage = allocation_percentage
        self.buy_price = None
        self.sell_price = None
        self.initial_shares = None
        self.additional_shares = 0
        self.total_shares = None
        self.investment_amount = None
        self.additional_investment = 0
        self.total_investment = None
        self.profit_loss = None
        self.profit_loss_percentage = None
        self.dip_buys = []  # List to track dip buy points
        self.momentum_buys = []  # List to track momentum buy points

    def execute(self, initial_balance: float) -> None:
        # Get historical data
        stock = yf.Ticker(self.ticker)
        # Get data with extra days for indicators
        start_date = self.buy_date - timedelta(days=100)
        hist = stock.history(start=start_date, end=self.sell_date)
        
        if hist.empty:
            raise ValueError(f"No data available for {self.ticker} between {self.buy_date} and {self.sell_date}")
        
        # Ensure index is timezone-naive
        hist.index = hist.index.tz_localize(None)
        
        # Calculate technical indicators
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = self.calculate_rsi(hist['Close'])
        
        # Calculate momentum indicators
        hist['Price_Change'] = hist['Close'].pct_change()
        hist['Volume_Change'] = hist['Volume'].pct_change()
        hist['Momentum'] = hist['Close'] - hist['Close'].shift(5)  # 5-day momentum
        
        # Set buy price as the first available closing price on or after the buy date
        buy_price_row = hist[hist.index >= self.buy_date]
        if buy_price_row.empty:
            raise ValueError(f"No available trading day for {self.ticker} on or after {self.buy_date}")
        self.buy_price = buy_price_row['Close'].iloc[0]
        self.investment_amount = initial_balance * (self.allocation_percentage / 100)
        self.initial_shares = self.investment_amount / self.buy_price
        
        # Initialize variables for dip buying and momentum buying
        lowest_price = self.buy_price
        highest_price = self.buy_price
        current_shares = self.initial_shares
        current_investment = self.investment_amount
        last_dip_buy_date = None
        last_momentum_buy_date = None
        
        # Check each day's price action
        for i in range(1, len(hist)):
            current_price = hist['Close'].iloc[i]
            current_low = hist['Low'].iloc[i]
            current_high = hist['High'].iloc[i]
            current_date = hist.index[i]
            
            # Update highest and lowest prices
            highest_price = max(highest_price, current_high)
            lowest_price = min(lowest_price, current_low)
            
            # Calculate price changes
            price_change_from_high = (highest_price - current_price) / highest_price
            price_change_from_buy = (current_price - self.buy_price) / self.buy_price
            
            # Conditions for dip buying:
            # 1. Price is below 20-day SMA (downtrend)
            # 2. Price has dropped at least 10% from recent high
            # 3. 20-day SMA is still above buy price (overall uptrend)
            # 4. We haven't already made 2 additional buys
            # 5. Current price is below our initial buy price
            # 6. At least 5 trading days since last dip buy
            if (i >= 20 and
                current_price < hist['SMA20'].iloc[i] and 
                price_change_from_high >= 0.10 and 
                hist['SMA20'].iloc[i] > self.buy_price and 
                len(self.dip_buys) < 2 and  # Allow up to 2 additional buys
                current_price < self.buy_price and  # Only buy if price is below initial buy price
                (last_dip_buy_date is None or (current_date - last_dip_buy_date).days >= 5)):  # 5-day gap
                
                # Calculate additional investment (same as initial)
                additional_investment = self.investment_amount
                additional_shares = additional_investment / current_price
                
                # Update position
                self.additional_shares += additional_shares
                self.additional_investment += additional_investment
                current_shares += additional_shares
                current_investment += additional_investment
                
                # Record dip buy
                self.dip_buys.append({
                    'date': current_date,
                    'price': current_price,
                    'shares': additional_shares,
                    'investment': additional_investment,
                    'type': 'dip'
                })
                last_dip_buy_date = current_date
            
            # Conditions for momentum buying:
            # 1. Price is above both 20-day and 50-day SMAs (strong uptrend)
            # 2. RSI is between 50-70 (not overbought)
            # 3. 5-day momentum is positive
            # 4. Volume is increasing (at least 20% higher than previous day)
            # 5. Price is at least 5% above initial buy price
            # 6. We haven't already made 2 momentum buys
            # 7. At least 5 trading days since last momentum buy
            elif (i >= 50 and
                  current_price > hist['SMA20'].iloc[i] and
                  current_price > hist['SMA50'].iloc[i] and
                  50 <= hist['RSI'].iloc[i] <= 70 and
                  hist['Momentum'].iloc[i] > 0 and
                  hist['Volume_Change'].iloc[i] > 0.20 and
                  price_change_from_buy >= 0.05 and
                  len(self.momentum_buys) < 2 and
                  (last_momentum_buy_date is None or (current_date - last_momentum_buy_date).days >= 5)):
                
                # Calculate additional investment (same as initial)
                additional_investment = self.investment_amount
                additional_shares = additional_investment / current_price
                
                # Update position
                self.additional_shares += additional_shares
                self.additional_investment += additional_investment
                current_shares += additional_shares
                current_investment += additional_investment
                
                # Record momentum buy
                self.momentum_buys.append({
                    'date': current_date,
                    'price': current_price,
                    'shares': additional_shares,
                    'investment': additional_investment,
                    'type': 'momentum'
                })
                last_momentum_buy_date = current_date
        
        # Calculate final position
        self.total_shares = self.initial_shares + self.additional_shares
        self.total_investment = self.investment_amount + self.additional_investment
        
        # Final sell at end date
        self.sell_price = hist['Close'].iloc[-1]
        self.profit_loss = (self.sell_price - self.buy_price) * self.initial_shares
        
        # Add profit/loss from additional shares (both dip and momentum buys)
        for additional_buy in self.dip_buys + self.momentum_buys:
            self.profit_loss += (self.sell_price - additional_buy['price']) * additional_buy['shares']
        
        # Calculate overall profit percentage
        self.profit_loss_percentage = (self.profit_loss / self.total_investment) * 100

    def calculate_rsi(self, prices, period=14):
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class Backtester:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades: List[Trade] = []
        self.results: List[Dict] = []

    def load_trades(self, csv_path: str) -> None:
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['ticker', 'sell_date', 'allocation_percentage']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            for _, row in df.iterrows():
                buy_date = row.get('buy_date', datetime.now().strftime('%Y-%m-%d'))
                trade = Trade(
                    ticker=row['ticker'],
                    buy_date=buy_date,
                    sell_date=row['sell_date'],
                    allocation_percentage=float(row['allocation_percentage'])
                )
                self.trades.append(trade)
        except Exception as e:
            print(f"Error loading trades: {str(e)}")
            sys.exit(1)

    def run_backtest(self) -> None:
        for trade in self.trades:
            try:
                trade.execute(self.current_balance)
                # Only add to results if we actually made the trade (shares > 0)
                if trade.initial_shares > 0:
                    self.current_balance += trade.profit_loss
                    
                    result = {
                        'ticker': trade.ticker,
                        'buy_date': trade.buy_date.strftime('%Y-%m-%d'),
                        'sell_date': trade.sell_date.strftime('%Y-%m-%d'),
                        'buy_price': round(trade.buy_price, 2),
                        'sell_price': round(trade.sell_price, 2),
                        'initial_shares': round(trade.initial_shares, 4),
                        'additional_shares': round(trade.additional_shares, 4),
                        'total_shares': round(trade.total_shares, 4),
                        'initial_investment': round(trade.investment_amount, 2),
                        'additional_investment': round(trade.additional_investment, 2),
                        'total_investment': round(trade.total_investment, 2),
                        'profit_loss': round(trade.profit_loss, 2),
                        'profit_loss_percentage': round(trade.profit_loss_percentage, 2),
                        'dip_buys': len(trade.dip_buys),
                        'momentum_buys': len(trade.momentum_buys)
                    }
                    self.results.append(result)
            except Exception as e:
                print(f"Error executing trade for {trade.ticker}: {str(e)}")

    def print_results(self, detailed: bool = False) -> None:
        if detailed:
            # Create a list to store all trade rows (including dip buys)
            all_trades = []
            
            for result in self.results:
                # Add initial trade
                initial_trade = {
                    'Ticker': result['ticker'],
                    'Buy Date': result['buy_date'],
                    'Sell Date': result['sell_date'],
                    'Buy Price': f"${result['buy_price']:,.2f}",
                    'Sell Price': f"${result['sell_price']:,.2f}",
                    'Shares': f"{result['initial_shares']:,.4f}",
                    'Investment': f"${result['initial_investment']:,.2f}",
                    'Profit/Loss': f"${(result['sell_price'] - result['buy_price']) * result['initial_shares']:,.2f}",
                    'Profit/Loss %': f"{((result['sell_price'] - result['buy_price']) / result['buy_price'] * 100):,.2f}%",
                    'Dip Buy': 'No',
                    'Momentum Buy': 'No'
                }
                all_trades.append(initial_trade)
                
                # Add dip buys if any
                if result['dip_buys'] > 0:
                    for dip_buy in self.trades[self.results.index(result)].dip_buys:
                        dip_trade = {
                            'Ticker': result['ticker'],
                            'Buy Date': dip_buy['date'].strftime('%Y-%m-%d'),
                            'Sell Date': result['sell_date'],
                            'Buy Price': f"${dip_buy['price']:,.2f}",
                            'Sell Price': f"${result['sell_price']:,.2f}",
                            'Shares': f"{dip_buy['shares']:,.4f}",
                            'Investment': f"${dip_buy['investment']:,.2f}",
                            'Profit/Loss': f"${(result['sell_price'] - dip_buy['price']) * dip_buy['shares']:,.2f}",
                            'Profit/Loss %': f"{((result['sell_price'] - dip_buy['price']) / dip_buy['price'] * 100):,.2f}%",
                            'Dip Buy': 'Yes',
                            'Momentum Buy': 'No'
                        }
                        all_trades.append(dip_trade)
                
                # Add momentum buys if any
                if result['momentum_buys'] > 0:
                    for momentum_buy in self.trades[self.results.index(result)].momentum_buys:
                        momentum_trade = {
                            'Ticker': result['ticker'],
                            'Buy Date': momentum_buy['date'].strftime('%Y-%m-%d'),
                            'Sell Date': result['sell_date'],
                            'Buy Price': f"${momentum_buy['price']:,.2f}",
                            'Sell Price': f"${result['sell_price']:,.2f}",
                            'Shares': f"{momentum_buy['shares']:,.4f}",
                            'Investment': f"${momentum_buy['investment']:,.2f}",
                            'Profit/Loss': f"${(result['sell_price'] - momentum_buy['price']) * momentum_buy['shares']:,.2f}",
                            'Profit/Loss %': f"{((result['sell_price'] - momentum_buy['price']) / momentum_buy['price'] * 100):,.2f}%",
                            'Dip Buy': 'No',
                            'Momentum Buy': 'Yes'
                        }
                        all_trades.append(momentum_trade)
            
            # Create DataFrame from all trades
            trades_df = pd.DataFrame(all_trades)
            
            # Print individual trade results
            print("\n=== Individual Trade Results ===")
            print(trades_df.to_string(index=False))
        
        # Calculate time-based metrics
        trades_df = pd.DataFrame(self.results)
        earliest_buy = pd.to_datetime(trades_df['buy_date'].min())
        latest_sell = pd.to_datetime(trades_df['sell_date'].max())
        
        # Calculate total days and months
        total_days = (latest_sell - earliest_buy).days
        total_months = total_days / 30.44  # Average days in a month
        
        # Calculate returns
        total_return = self.current_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        # Calculate periodic returns
        monthly_return_pct = total_return_pct / total_months
        quarterly_return_pct = monthly_return_pct * 3
        yearly_return_pct = monthly_return_pct * 12
        
        # Calculate trade statistics
        total_trades = len(trades_df)
        unique_symbols = len(trades_df['ticker'].unique())
        avg_profit_loss = trades_df['profit_loss'].mean()
        avg_profit_loss_pct = trades_df['profit_loss_percentage'].mean()
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate dip buying statistics
        total_dip_buys = trades_df['dip_buys'].sum()
        avg_dip_buys = trades_df['dip_buys'].mean()
        
        # Calculate momentum buying statistics
        total_momentum_buys = trades_df['momentum_buys'].sum()
        avg_momentum_buys = trades_df['momentum_buys'].mean()
        
        # Print summary
        print("\n=== Backtesting Summary ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: ${total_return:,.2f}")
        print(f"Total Return Percentage: {total_return_pct:,.2f}%")
        
        print("\n=== Trade Statistics ===")
        print(f"Total Trades: {total_trades}")
        print(f"Unique Symbols: {unique_symbols}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Profit/Loss per Trade: ${avg_profit_loss:,.2f}")
        print(f"Average Profit/Loss % per Trade: {avg_profit_loss_pct:,.2f}%")
        print(f"Total Dip Buys: {total_dip_buys}")
        print(f"Average Dip Buys per Trade: {avg_dip_buys:.1f}")
        print(f"Total Momentum Buys: {total_momentum_buys}")
        print(f"Average Momentum Buys per Trade: {avg_momentum_buys:.1f}")
        
        print("\n=== Time Period Analysis ===")
        print(f"Earliest Buy Date: {earliest_buy.strftime('%Y-%m-%d')}")
        print(f"Latest Sell Date: {latest_sell.strftime('%Y-%m-%d')}")
        print(f"Total Days: {total_days}")
        print(f"Total Months: {total_months:.1f}")
        print(f"Monthly Return: {monthly_return_pct:,.2f}%")
        print(f"Quarterly Return: {quarterly_return_pct:,.2f}%")
        print(f"Annual Return: {yearly_return_pct:,.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Run backtesting on trading strategies')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--detailed', action='store_true', help='Show detailed trade results')
    
    args = parser.parse_args()
    
    backtester = Backtester(initial_balance=args.initial_balance)
    backtester.load_trades(args.input)
    backtester.run_backtest()
    backtester.print_results(detailed=args.detailed)

if __name__ == "__main__":
    main() 