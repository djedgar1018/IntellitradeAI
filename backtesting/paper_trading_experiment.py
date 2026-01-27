"""
Paper Trading Experiment - 6 Month Backtesting
===============================================
Three accounts: $10,000 each for Stocks, Crypto, and Forex
Max drawdown threshold: 40%
Goal: Maximize account growth using AI trading signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, using simulated data")


class TradingSignalGenerator:
    """Generate trading signals based on technical indicators"""
    
    def __init__(self):
        self.lookback_period = 20
        
    def calculate_indicators(self, df):
        """Calculate technical indicators for signal generation"""
        df = df.copy()
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # ATR for position sizing
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Momentum
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        df['momentum_pct'] = df['Close'].pct_change(periods=10) * 100
        
        return df
    
    def generate_signal(self, df, idx):
        """Generate trading signal for a specific date"""
        if idx < 50:  # Need enough data for indicators
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        buy_signals = 0
        sell_signals = 0
        reasons = []
        
        # SMA crossover
        if pd.notna(row['sma_10']) and pd.notna(row['sma_20']):
            if row['sma_10'] > row['sma_20'] and prev_row['sma_10'] <= prev_row['sma_20']:
                buy_signals += 2
                reasons.append("SMA 10/20 bullish crossover")
            elif row['sma_10'] < row['sma_20'] and prev_row['sma_10'] >= prev_row['sma_20']:
                sell_signals += 2
                reasons.append("SMA 10/20 bearish crossover")
            elif row['sma_10'] > row['sma_20']:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # MACD
        if pd.notna(row['macd']) and pd.notna(row['macd_signal']):
            if row['macd'] > row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']:
                buy_signals += 2
                reasons.append("MACD bullish crossover")
            elif row['macd'] < row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']:
                sell_signals += 2
                reasons.append("MACD bearish crossover")
            elif row['macd'] > row['macd_signal']:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # RSI
        if pd.notna(row['rsi']):
            if row['rsi'] < 30:
                buy_signals += 2
                reasons.append(f"RSI oversold ({row['rsi']:.1f})")
            elif row['rsi'] > 70:
                sell_signals += 2
                reasons.append(f"RSI overbought ({row['rsi']:.1f})")
            elif row['rsi'] < 40:
                buy_signals += 1
            elif row['rsi'] > 60:
                sell_signals += 1
        
        # Bollinger Bands
        if pd.notna(row['bb_lower']) and pd.notna(row['bb_upper']):
            if row['Close'] < row['bb_lower']:
                buy_signals += 2
                reasons.append("Price below lower Bollinger Band")
            elif row['Close'] > row['bb_upper']:
                sell_signals += 2
                reasons.append("Price above upper Bollinger Band")
        
        # Momentum
        if pd.notna(row['momentum_pct']):
            if row['momentum_pct'] > 5:
                buy_signals += 1
            elif row['momentum_pct'] < -5:
                sell_signals += 1
        
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No clear signals'}
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.65:
            confidence = min(0.95, buy_ratio)
            return {
                'signal': 'BUY',
                'confidence': confidence,
                'reason': '; '.join(reasons) if reasons else 'Multiple bullish indicators'
            }
        elif sell_ratio > 0.65:
            confidence = min(0.95, sell_ratio)
            return {
                'signal': 'SELL',
                'confidence': confidence,
                'reason': '; '.join(reasons) if reasons else 'Multiple bearish indicators'
            }
        else:
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Mixed signals'}


class Position:
    """Represents a trading position"""
    def __init__(self, symbol, entry_price, shares, position_type, entry_date, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.position_type = position_type  # 'long' or 'short'
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0
        self.pnl_pct = 0
        
    def close(self, exit_price, exit_date):
        self.exit_price = exit_price
        self.exit_date = exit_date
        if self.position_type == 'long':
            self.pnl = (exit_price - self.entry_price) * self.shares
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.shares
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100
        return self.pnl


class PaperTradingAccount:
    """Paper trading account with full tracking"""
    
    def __init__(self, name, initial_balance, max_drawdown_pct=40):
        self.name = name
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.positions = {}  # symbol -> Position
        self.closed_trades = []
        self.equity_curve = []
        self.peak_equity = initial_balance
        self.max_drawdown = 0
        self.failed = False
        self.fail_reason = None
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        
    def get_equity(self, current_prices):
        """Calculate current equity including open positions"""
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if position.position_type == 'long':
                    equity += position.shares * current_price
                else:
                    equity += position.shares * (2 * position.entry_price - current_price)
        return equity
    
    def update_equity_curve(self, date, current_prices):
        """Update equity curve and check drawdown"""
        equity = self.get_equity(current_prices)
        self.equity_curve.append({'date': date, 'equity': equity})
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if drawdown >= self.max_drawdown_pct:
            self.failed = True
            self.fail_reason = f"Max drawdown exceeded: {drawdown:.2f}%"
            return False
        return True
    
    def open_position(self, symbol, price, position_type, date, risk_pct=2):
        """Open a new position with proper position sizing"""
        if symbol in self.positions:
            return False  # Already have a position
        
        # Position sizing based on risk
        risk_amount = self.balance * (risk_pct / 100)
        stop_distance = price * 0.05  # 5% stop loss
        shares = risk_amount / stop_distance
        position_value = shares * price
        
        # Limit to 20% of account per position
        max_position = self.balance * 0.20
        if position_value > max_position:
            shares = max_position / price
            position_value = shares * price
        
        if position_value > self.balance:
            shares = self.balance * 0.95 / price
            position_value = shares * price
        
        if position_value < 100:  # Minimum position size
            return False
        
        stop_loss = price * 0.95 if position_type == 'long' else price * 1.05
        take_profit = price * 1.10 if position_type == 'long' else price * 0.90
        
        position = Position(symbol, price, shares, position_type, date, stop_loss, take_profit)
        self.positions[symbol] = position
        self.balance -= position_value
        return True
    
    def close_position(self, symbol, price, date, reason='signal'):
        """Close an existing position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        pnl = position.close(price, date)
        
        # Update balance
        if position.position_type == 'long':
            self.balance += position.shares * price
        else:
            self.balance += position.shares * (2 * position.entry_price - price)
        
        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        position.close_reason = reason
        self.closed_trades.append(position)
        del self.positions[symbol]
        return pnl
    
    def check_stops(self, symbol, high, low, date):
        """Check if stop loss or take profit hit"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position.position_type == 'long':
            if low <= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, date, 'stop_loss')
            if high >= position.take_profit:
                return self.close_position(symbol, position.take_profit, date, 'take_profit')
        else:
            if high >= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, date, 'stop_loss')
            if low <= position.take_profit:
                return self.close_position(symbol, position.take_profit, date, 'take_profit')
        return None
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.equity_curve:
            return {}
        
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity / self.initial_balance - 1) * 100
        
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if len(equity_df) > 1 else 0
        
        return {
            'account_name': self.name,
            'initial_balance': self.initial_balance,
            'final_equity': round(final_equity, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
            'max_drawdown_pct': round(self.max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2) if not np.isnan(sharpe) else 0,
            'failed': self.failed,
            'fail_reason': self.fail_reason
        }


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self):
        self.signal_generator = TradingSignalGenerator()
        
    def fetch_data(self, symbol, start_date, end_date):
        """Fetch historical data"""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                if len(df) > 0:
                    return df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        # Generate simulated data if yfinance fails
        return self.generate_simulated_data(symbol, start_date, end_date)
    
    def generate_simulated_data(self, symbol, start_date, end_date):
        """Generate realistic simulated price data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Asset-specific base prices
        base_prices = {
            'AAPL': 180, 'MSFT': 400, 'GOOGL': 140, 'AMZN': 180, 'NVDA': 500,
            'TSLA': 250, 'META': 500, 'AMD': 150, 'NFLX': 600, 'JPM': 200,
            'BTC-USD': 45000, 'ETH-USD': 2500, 'SOL-USD': 100, 'XRP-USD': 0.60, 'ADA-USD': 0.50,
            'DOGE-USD': 0.08, 'AVAX-USD': 35, 'DOT-USD': 7, 'MATIC-USD': 0.80, 'LINK-USD': 15,
            'EURUSD=X': 1.08, 'GBPUSD=X': 1.27, 'USDJPY=X': 150, 'AUDUSD=X': 0.65, 'USDCAD=X': 1.35
        }
        
        base_price = base_prices.get(symbol, 100)
        volatility = 0.02 if 'USD' in symbol and '=' in symbol else 0.03 if '-USD' in symbol else 0.02
        
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0003, volatility, len(dates))
        
        # Add some trend and mean reversion
        trend = np.linspace(0, 0.1, len(dates)) * (1 if np.random.random() > 0.5 else -1)
        returns = returns + trend / len(dates)
        
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC
        daily_range = prices * np.random.uniform(0.01, 0.03, len(dates))
        opens = prices * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
        highs = np.maximum(opens, prices) + daily_range * 0.5
        lows = np.minimum(opens, prices) - daily_range * 0.5
        volumes = np.random.randint(1000000, 50000000, len(dates))
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return df
    
    def run_backtest(self, account, symbols, start_date, end_date):
        """Run backtest for a list of symbols"""
        print(f"\n{'='*60}")
        print(f"Running backtest for: {account.name}")
        print(f"Symbols: {symbols}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}")
        
        # Fetch and prepare data for all symbols
        all_data = {}
        for symbol in symbols:
            df = self.fetch_data(symbol, start_date, end_date)
            if df is not None and len(df) > 50:
                df = self.signal_generator.calculate_indicators(df)
                all_data[symbol] = df
                print(f"Loaded {len(df)} days of data for {symbol}")
        
        if not all_data:
            print("No data available for backtesting")
            return
        
        # Get all trading days
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        trading_days = sorted(all_dates)
        
        # Run simulation
        for i, date in enumerate(trading_days):
            if account.failed:
                break
            
            current_prices = {}
            
            for symbol, df in all_data.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                row = df.iloc[idx]
                current_prices[symbol] = row['Close']
                
                # Check stops first
                account.check_stops(symbol, row['High'], row['Low'], date)
                
                # Generate signal
                signal_result = self.signal_generator.generate_signal(df, idx)
                
                # Execute trades based on signals
                if signal_result['signal'] == 'BUY' and signal_result['confidence'] > 0.6:
                    if symbol not in account.positions:
                        if account.open_position(symbol, row['Close'], 'long', date):
                            print(f"  {date.date()}: BUY {symbol} @ ${row['Close']:.2f} (conf: {signal_result['confidence']:.2f})")
                
                elif signal_result['signal'] == 'SELL' and signal_result['confidence'] > 0.6:
                    if symbol in account.positions:
                        pnl = account.close_position(symbol, row['Close'], date, 'sell_signal')
                        if pnl is not None:
                            print(f"  {date.date()}: SELL {symbol} @ ${row['Close']:.2f} (PnL: ${pnl:.2f})")
            
            # Update equity curve
            if current_prices:
                if not account.update_equity_curve(date, current_prices):
                    print(f"\n*** ACCOUNT FAILED: {account.fail_reason} ***")
                    break
        
        # Close any remaining positions at end
        for symbol in list(account.positions.keys()):
            if symbol in all_data and len(all_data[symbol]) > 0:
                final_price = all_data[symbol]['Close'].iloc[-1]
                account.close_position(symbol, final_price, trading_days[-1], 'end_of_period')
        
        return account.get_performance_summary()


def run_paper_trading_experiment():
    """Run the full paper trading experiment"""
    
    # Define time period (6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Define assets for each category
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'JPM']
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD']
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
    
    # Create accounts
    stock_account = PaperTradingAccount("Stocks Portfolio", 10000)
    crypto_account = PaperTradingAccount("Crypto Portfolio", 10000)
    forex_account = PaperTradingAccount("Forex Portfolio", 10000)
    
    # Run backtests
    engine = BacktestEngine()
    
    results = {}
    results['stocks'] = engine.run_backtest(stock_account, stock_symbols, start_date, end_date)
    results['crypto'] = engine.run_backtest(crypto_account, crypto_symbols, start_date, end_date)
    results['forex'] = engine.run_backtest(forex_account, forex_symbols, start_date, end_date)
    
    # Generate report
    print("\n" + "="*80)
    print("PAPER TRADING EXPERIMENT RESULTS")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()} (6 months)")
    print(f"Initial Capital: $10,000 per portfolio ($30,000 total)")
    print(f"Max Drawdown Threshold: 40%")
    print("="*80)
    
    total_final = 0
    total_trades = 0
    total_wins = 0
    
    for name, summary in results.items():
        if summary:
            print(f"\n{'-'*40}")
            print(f"{summary['account_name'].upper()}")
            print(f"{'-'*40}")
            print(f"  Final Equity:     ${summary['final_equity']:,.2f}")
            print(f"  Total Return:     {summary['total_return_pct']:+.2f}%")
            print(f"  Total Trades:     {summary['total_trades']}")
            print(f"  Win Rate:         {summary['win_rate_pct']:.1f}%")
            print(f"  Profit Factor:    {summary['profit_factor']}")
            print(f"  Max Drawdown:     {summary['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe Ratio:     {summary['sharpe_ratio']:.2f}")
            if summary['failed']:
                print(f"  *** FAILED: {summary['fail_reason']} ***")
            
            total_final += summary['final_equity']
            total_trades += summary['total_trades']
            total_wins += summary['winning_trades']
    
    print(f"\n{'='*80}")
    print("COMBINED RESULTS")
    print(f"{'='*80}")
    print(f"  Total Final Equity:  ${total_final:,.2f}")
    print(f"  Total Return:        {((total_final / 30000) - 1) * 100:+.2f}%")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Overall Win Rate:    {(total_wins / total_trades * 100) if total_trades > 0 else 0:.1f}%")
    
    # Generate improvement recommendations
    print(f"\n{'='*80}")
    print("STRATEGY IMPROVEMENT RECOMMENDATIONS")
    print(f"{'='*80}")
    
    improvements = []
    
    for name, summary in results.items():
        if summary:
            if summary['win_rate_pct'] < 50:
                improvements.append(f"- {name.upper()}: Low win rate ({summary['win_rate_pct']:.1f}%) - Consider tightening entry criteria")
            if summary['max_drawdown_pct'] > 25:
                improvements.append(f"- {name.upper()}: High drawdown ({summary['max_drawdown_pct']:.1f}%) - Reduce position sizes")
            if summary['profit_factor'] != 'N/A' and summary['profit_factor'] < 1.5:
                improvements.append(f"- {name.upper()}: Low profit factor ({summary['profit_factor']}) - Improve risk/reward ratio")
            if summary['failed']:
                improvements.append(f"- {name.upper()}: CRITICAL - Strategy failed with {summary['max_drawdown_pct']:.1f}% drawdown")
    
    if not improvements:
        improvements.append("- All strategies performing within acceptable parameters")
    
    for imp in improvements:
        print(imp)
    
    print("\nGENERAL RECOMMENDATIONS:")
    print("1. Add trend filter to avoid trading against major trend")
    print("2. Implement dynamic position sizing based on volatility")
    print("3. Add correlation analysis to avoid concentrated risk")
    print("4. Consider adding fundamental filters for stocks")
    print("5. Implement trailing stops to lock in profits")
    
    # Save results to file
    results_file = 'backtesting/experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'run_date': datetime.now().isoformat(),
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'results': results,
            'improvements': improvements
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_paper_trading_experiment()
