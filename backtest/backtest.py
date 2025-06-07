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
        self.partial_sells = []  # List to track partial sells
        self.current_shares = 0  # Track current position size

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
        
        # Calculate volatility indicators
        hist['ATR'] = self.calculate_atr(hist, period=14)
        hist['Volatility'] = hist['ATR'] / hist['Close'] * 100  # Volatility as percentage
        
        # Set buy price as the first available closing price on or after the buy date
        buy_price_row = hist[hist.index >= self.buy_date]
        if buy_price_row.empty:
            raise ValueError(f"No available trading day for {self.ticker} on or after {self.buy_date}")
        self.buy_price = buy_price_row['Close'].iloc[0]
        self.investment_amount = initial_balance * (self.allocation_percentage / 100)
        self.initial_shares = self.investment_amount / self.buy_price
        self.current_shares = self.initial_shares
        
        # Initialize variables for dip buying and momentum buying
        lowest_price = self.buy_price
        highest_price = self.buy_price
        current_investment = self.investment_amount
        last_dip_buy_date = None
        last_momentum_buy_date = None
        last_partial_sell_date = None
        
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
            
            # Check for partial selling opportunities
            if self.current_shares > 0 and (last_partial_sell_date is None or (current_date - last_partial_sell_date).days >= 5):
                # 1. Take profit on significant gains (30% or more)
                if price_change_from_buy >= 0.30:
                    sell_percentage = 0.50  # Sell 50% of position
                    self._execute_partial_sell(current_date, current_price, sell_percentage, "take_profit")
                    last_partial_sell_date = current_date
                
                # 2. Cut losses on significant downtrend
                elif (current_price < hist['SMA20'].iloc[i] and 
                      current_price < hist['SMA50'].iloc[i] and 
                      hist['RSI'].iloc[i] < 30 and 
                      price_change_from_buy <= -0.15):  # 15% loss
                    sell_percentage = 0.75  # Sell 75% of position
                    self._execute_partial_sell(current_date, current_price, sell_percentage, "cut_loss")
                    last_partial_sell_date = current_date
                
                # 3. Reduce position on overbought conditions
                elif (hist['RSI'].iloc[i] > 80 and 
                      current_price > hist['SMA20'].iloc[i] * 1.1 and  # 10% above SMA20
                      hist['Volume_Change'].iloc[i] < -0.20):  # Declining volume
                    sell_percentage = 0.25  # Sell 25% of position
                    self._execute_partial_sell(current_date, current_price, sell_percentage, "overbought")
                    last_partial_sell_date = current_date
                
                # 4. Reduce position on high volatility
                elif (hist['Volatility'].iloc[i] > 5.0 and  # 5% daily volatility
                      price_change_from_buy > 0.20):  # In profit
                    sell_percentage = 0.33  # Sell 33% of position
                    self._execute_partial_sell(current_date, current_price, sell_percentage, "volatility")
                    last_partial_sell_date = current_date
            
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
                self.current_shares += additional_shares
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
                self.current_shares += additional_shares
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
        self.total_shares = self.current_shares  # Use current_shares instead of initial + additional
        self.total_investment = self.investment_amount + self.additional_investment
        
        # Final sell at end date
        self.sell_price = hist['Close'].iloc[-1]
        
        # Calculate profit/loss including partial sells
        self.profit_loss = 0
        
        # Add profit/loss from initial position
        if self.initial_shares > 0:
            self.profit_loss += (self.sell_price - self.buy_price) * self.initial_shares
        
        # Add profit/loss from dip buys
        for dip_buy in self.dip_buys:
            self.profit_loss += (self.sell_price - dip_buy['price']) * dip_buy['shares']
        
        # Add profit/loss from momentum buys
        for momentum_buy in self.momentum_buys:
            self.profit_loss += (self.sell_price - momentum_buy['price']) * momentum_buy['shares']
        
        # Add profit/loss from partial sells
        for partial_sell in self.partial_sells:
            self.profit_loss += (partial_sell['price'] - partial_sell['basis_price']) * partial_sell['shares']
        
        # Calculate overall profit percentage
        self.profit_loss_percentage = (self.profit_loss / self.total_investment) * 100

    def _execute_partial_sell(self, date, price, percentage, reason):
        """Execute a partial sell of the current position."""
        shares_to_sell = self.current_shares * percentage
        if shares_to_sell > 0:
            # Calculate average basis price for the shares being sold
            basis_price = self._calculate_basis_price(shares_to_sell)
            
            self.partial_sells.append({
                'date': date,
                'price': price,
                'shares': shares_to_sell,
                'basis_price': basis_price,
                'reason': reason,
                'profit_at_sell': (price - basis_price) * shares_to_sell,
                'profit_pct_at_sell': ((price - basis_price) / basis_price) * 100
            })
            
            # Update current position
            self.current_shares -= shares_to_sell

    def _calculate_basis_price(self, shares_to_sell):
        """Calculate the average basis price for the shares being sold."""
        total_cost = 0
        remaining_shares = shares_to_sell
        
        # First use initial position
        if self.initial_shares > 0:
            shares_from_initial = min(remaining_shares, self.initial_shares)
            total_cost += shares_from_initial * self.buy_price
            remaining_shares -= shares_from_initial
        
        # Then use dip buys
        for dip_buy in self.dip_buys:
            if remaining_shares <= 0:
                break
            shares_from_dip = min(remaining_shares, dip_buy['shares'])
            total_cost += shares_from_dip * dip_buy['price']
            remaining_shares -= shares_from_dip
        
        # Finally use momentum buys
        for momentum_buy in self.momentum_buys:
            if remaining_shares <= 0:
                break
            shares_from_momentum = min(remaining_shares, momentum_buy['shares'])
            total_cost += shares_from_momentum * momentum_buy['price']
            remaining_shares -= shares_from_momentum
        
        return total_cost / shares_to_sell if shares_to_sell > 0 else 0

    def calculate_atr(self, hist, period=14):
        """Calculate Average True Range (ATR)."""
        high = hist['High']
        low = hist['Low']
        close = hist['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

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
                        'momentum_buys': len(trade.momentum_buys),
                        'partial_sells': len(trade.partial_sells) if hasattr(trade, 'partial_sells') else 0
                    }
                    self.results.append(result)
            except Exception as e:
                print(f"Error executing trade for {trade.ticker}: {str(e)}")

    def print_results(self, detailed: bool = False) -> None:
        if detailed:
            # Create a list to store all trade rows (including dip buys and partial sells)
            all_trades = []
            
            for result in self.results:
                trade = self.trades[self.results.index(result)]
                
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
                    'Momentum Buy': 'No',
                    'Partial Sell': 'No'
                }
                all_trades.append(initial_trade)
                
                # Add dip buys if any
                if result['dip_buys'] > 0:
                    for dip_buy in trade.dip_buys:
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
                            'Momentum Buy': 'No',
                            'Partial Sell': 'No'
                        }
                        all_trades.append(dip_trade)
                
                # Add momentum buys if any
                if result['momentum_buys'] > 0:
                    for momentum_buy in trade.momentum_buys:
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
                            'Momentum Buy': 'Yes',
                            'Partial Sell': 'No'
                        }
                        all_trades.append(momentum_trade)
                
                # Add partial sells if any
                if hasattr(trade, 'partial_sells') and trade.partial_sells:
                    for partial_sell in trade.partial_sells:
                        # Calculate what would have happened if held until end date
                        # For the shares that were sold, what would they be worth at end date
                        end_date_value = (result['sell_price'] - partial_sell['basis_price']) * partial_sell['shares']
                        end_date_pct = ((result['sell_price'] - partial_sell['basis_price']) / partial_sell['basis_price']) * 100
                        
                        # Calculate the difference between actual sell and end date
                        value_difference = partial_sell['profit_at_sell'] - end_date_value
                        pct_difference = partial_sell['profit_pct_at_sell'] - end_date_pct
                        
                        # Calculate the actual profit from selling early
                        actual_profit = partial_sell['profit_at_sell']
                        actual_pct = partial_sell['profit_pct_at_sell']
                        
                        partial_sell_trade = {
                            'Ticker': result['ticker'],
                            'Buy Date': 'N/A',
                            'Sell Date': partial_sell['date'].strftime('%Y-%m-%d'),
                            'Buy Price': f"${partial_sell['basis_price']:,.2f}",
                            'Sell Price': f"${partial_sell['price']:,.2f}",
                            'Shares': f"{partial_sell['shares']:,.4f}",
                            'Investment': f"${partial_sell['shares'] * partial_sell['basis_price']:,.2f}",
                            'Actual Profit': f"${actual_profit:,.2f}",
                            'Actual Return %': f"{actual_pct:,.2f}%",
                            'End Date Value': f"${end_date_value:,.2f}",
                            'End Date Return %': f"{end_date_pct:,.2f}%",
                            'Value Difference': f"${value_difference:,.2f}",
                            'Return Difference %': f"{pct_difference:,.2f}%",
                            'Dip Buy': 'No',
                            'Momentum Buy': 'No',
                            'Partial Sell': f"Yes ({partial_sell['reason']})"
                        }
                        all_trades.append(partial_sell_trade)
            
            # Create DataFrame from all trades
            trades_df = pd.DataFrame(all_trades)
            
            # Print individual trade results
            print("\n=== Individual Trade Results ===")
            print(trades_df.to_string(index=False))
            
            # Calculate and print partial sell summary
            partial_sells = trades_df[trades_df['Partial Sell'].str.contains('Yes', na=False)]
            
            if not partial_sells.empty:
                print("\n=== Partial Sell Analysis ===")
                
                # Calculate totals
                total_actual_profit = sum(float(x.strip('$').replace(',', '')) for x in partial_sells['Actual Profit'])
                total_end_date_value = sum(float(x.strip('$').replace(',', '')) for x in partial_sells['End Date Value'])
                total_investment = sum(float(x.strip('$').replace(',', '')) for x in partial_sells['Investment'])
                
                # Calculate return percentages
                actual_return_pct = (total_actual_profit / total_investment) * 100 if total_investment > 0 else 0
                end_date_return_pct = (total_end_date_value / total_investment) * 100 if total_investment > 0 else 0
                return_difference_pct = actual_return_pct - end_date_return_pct
                
                print("Overall Performance:")
                print(f"1. Total Investment in Partial Sells: ${total_investment:,.2f}")
                print(f"2. Total Profit from Partial Sells: ${total_actual_profit:,.2f}")
                print(f"3. Total Value if Held Until End Date: ${total_end_date_value:,.2f}")
                print(f"\nReturn Analysis:")
                print(f"1. Actual Return: {actual_return_pct:,.2f}%")
                print(f"2. End Date Return: {end_date_return_pct:,.2f}%")
                if return_difference_pct > 0:
                    print(f"3. Additional Return from Selling Early: {return_difference_pct:,.2f}%")
                else:
                    print(f"3. Return Lost by Selling Early: {abs(return_difference_pct):,.2f}%")
                
                # Break down by reason
                print("\nBreakdown by Sell Reason:")
                for reason in ['take_profit', 'cut_loss', 'overbought', 'volatility']:
                    reason_sells = partial_sells[partial_sells['Partial Sell'].str.contains(reason, na=False)]
                    if not reason_sells.empty:
                        reason_profit = sum(float(x.strip('$').replace(',', '')) for x in reason_sells['Actual Profit'])
                        reason_end_value = sum(float(x.strip('$').replace(',', '')) for x in reason_sells['End Date Value'])
                        reason_investment = sum(float(x.strip('$').replace(',', '')) for x in reason_sells['Investment'])
                        
                        # Calculate return percentages for this reason
                        reason_return_pct = (reason_profit / reason_investment) * 100 if reason_investment > 0 else 0
                        reason_end_return_pct = (reason_end_value / reason_investment) * 100 if reason_investment > 0 else 0
                        reason_return_diff_pct = reason_return_pct - reason_end_return_pct
                        
                        print(f"\n{reason.replace('_', ' ').title()}:")
                        print(f"  Number of Sells: {len(reason_sells)}")
                        print(f"  Total Investment: ${reason_investment:,.2f}")
                        print(f"  Total Profit from Sells: ${reason_profit:,.2f}")
                        print(f"  Value if Held Until End: ${reason_end_value:,.2f}")
                        print(f"\n  Return Analysis:")
                        print(f"  - Actual Return: {reason_return_pct:,.2f}%")
                        print(f"  - End Date Return: {reason_end_return_pct:,.2f}%")
                        if reason_return_diff_pct > 0:
                            print(f"  - Additional Return from Selling Early: {reason_return_diff_pct:,.2f}%")
                        else:
                            print(f"  - Return Lost by Selling Early: {abs(reason_return_diff_pct):,.2f}%")
                        
                        # Calculate average metrics
                        avg_profit = reason_profit / len(reason_sells)
                        avg_end_value = reason_end_value / len(reason_sells)
                        avg_investment = reason_investment / len(reason_sells)
                        
                        print(f"\n  Average Metrics per Sell:")
                        print(f"  - Average Investment: ${avg_investment:,.2f}")
                        print(f"  - Average Profit: ${avg_profit:,.2f}")
                        print(f"  - Average End Value: ${avg_end_value:,.2f}")
                        print(f"  - Average Return: {(avg_profit / avg_investment * 100):,.2f}%")
                        print(f"  - Average End Return: {(avg_end_value / avg_investment * 100):,.2f}%")
        
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
        
        # Calculate partial sell statistics
        total_partial_sells = sum(len(trade.partial_sells) for trade in self.trades)
        avg_partial_sells = total_partial_sells / len(self.trades) if self.trades else 0
        
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
        print(f"Total Partial Sells: {total_partial_sells}")
        print(f"Average Partial Sells per Trade: {avg_partial_sells:.1f}")
        
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