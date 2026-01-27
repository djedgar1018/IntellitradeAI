"""
Paper Trading Experiment V2 - IMPROVED STRATEGY
================================================
Implements key improvements from analysis:
1. Trend filter (SMA 50)
2. Higher confidence thresholds
3. Dynamic ATR-based stops
4. Trailing stops
5. Volume confirmation
6. Reduced trading frequency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, using simulated data")


class ImprovedSignalGenerator:
    """Enhanced signal generator with trend filter and volume confirmation"""
    
    def __init__(self, asset_class='stocks'):
        self.asset_class = asset_class
        self.confidence_thresholds = {
            'stocks': 0.70,
            'crypto': 0.75,
            'forex': 0.72
        }
        
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200, min_periods=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['Close'] * 100
        
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        df['momentum_pct'] = df['Close'].pct_change(periods=10) * 100
        
        df['trend'] = np.where(df['Close'] > df['sma_50'], 1, -1)
        
        return df
    
    def generate_signal(self, df, idx):
        """Generate signal with trend filter and volume confirmation"""
        if idx < 60:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data', 'atr': None}
        
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        
        in_uptrend = pd.notna(row['sma_50']) and row['Close'] > row['sma_50']
        in_downtrend = pd.notna(row['sma_50']) and row['Close'] < row['sma_50']
        
        volume_confirmed = pd.notna(row['volume_ratio']) and row['volume_ratio'] > 1.2
        
        buy_signals = 0
        sell_signals = 0
        reasons = []
        
        if pd.notna(row['sma_10']) and pd.notna(row['sma_20']):
            if row['sma_10'] > row['sma_20'] and prev_row['sma_10'] <= prev_row['sma_20']:
                buy_signals += 3
                reasons.append("SMA 10/20 bullish crossover")
            elif row['sma_10'] < row['sma_20'] and prev_row['sma_10'] >= prev_row['sma_20']:
                sell_signals += 3
                reasons.append("SMA 10/20 bearish crossover")
            elif row['sma_10'] > row['sma_20']:
                buy_signals += 1
            else:
                sell_signals += 1
        
        if pd.notna(row['macd']) and pd.notna(row['macd_signal']):
            if row['macd'] > row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']:
                buy_signals += 3
                reasons.append("MACD bullish crossover")
            elif row['macd'] < row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']:
                sell_signals += 3
                reasons.append("MACD bearish crossover")
            elif row['macd'] > 0 and row['macd'] > row['macd_signal']:
                buy_signals += 1
            elif row['macd'] < 0 and row['macd'] < row['macd_signal']:
                sell_signals += 1
        
        rsi_buy_threshold = 25 if self.asset_class == 'crypto' else 30
        rsi_sell_threshold = 75 if self.asset_class == 'crypto' else 70
        
        if pd.notna(row['rsi']):
            if row['rsi'] < rsi_buy_threshold:
                buy_signals += 3
                reasons.append(f"RSI oversold ({row['rsi']:.1f})")
            elif row['rsi'] > rsi_sell_threshold:
                sell_signals += 3
                reasons.append(f"RSI overbought ({row['rsi']:.1f})")
            elif row['rsi'] < 35:
                buy_signals += 1
            elif row['rsi'] > 65:
                sell_signals += 1
        
        if pd.notna(row['bb_lower']) and pd.notna(row['bb_upper']):
            bb_pct = (row['Close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if bb_pct < 0.1:
                buy_signals += 2
                reasons.append("Price near lower Bollinger Band")
            elif bb_pct > 0.9:
                sell_signals += 2
                reasons.append("Price near upper Bollinger Band")
        
        if pd.notna(row['momentum_pct']):
            if row['momentum_pct'] > 8:
                buy_signals += 1
            elif row['momentum_pct'] < -8:
                sell_signals += 1
        
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No clear signals', 'atr': row.get('atr')}
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        confidence_threshold = self.confidence_thresholds.get(self.asset_class, 0.70)
        
        if buy_ratio > 0.60:
            if not in_uptrend:
                return {'signal': 'HOLD', 'confidence': buy_ratio * 0.5, 
                        'reason': 'Buy signal rejected - not in uptrend', 'atr': row.get('atr')}
            
            confidence = min(0.95, buy_ratio)
            if volume_confirmed:
                confidence = min(0.98, confidence * 1.1)
                reasons.append("Volume confirmed")
            
            if confidence >= confidence_threshold:
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'reason': '; '.join(reasons) if reasons else 'Multiple bullish indicators',
                    'atr': row.get('atr')
                }
        
        elif sell_ratio > 0.60:
            if not in_downtrend:
                return {'signal': 'HOLD', 'confidence': sell_ratio * 0.5,
                        'reason': 'Sell signal rejected - not in downtrend', 'atr': row.get('atr')}
            
            confidence = min(0.95, sell_ratio)
            if volume_confirmed:
                confidence = min(0.98, confidence * 1.1)
            
            if confidence >= confidence_threshold:
                return {
                    'signal': 'SELL',
                    'confidence': confidence,
                    'reason': '; '.join(reasons) if reasons else 'Multiple bearish indicators',
                    'atr': row.get('atr')
                }
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Mixed signals or below threshold', 'atr': row.get('atr')}


class Position:
    """Position with trailing stop support"""
    def __init__(self, symbol, entry_price, shares, position_type, entry_date, stop_loss=None, take_profit=None, atr=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.position_type = position_type
        self.entry_date = entry_date
        self.initial_stop = stop_loss
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr = atr
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.trailing_active = False
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0
        self.pnl_pct = 0
        
    def update_trailing_stop(self, current_price):
        """Update trailing stop as price moves favorably"""
        if self.position_type == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            profit_pct = (current_price / self.entry_price - 1) * 100
            
            if profit_pct >= 5:
                new_stop = self.entry_price * 1.02
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.trailing_active = True
            elif profit_pct >= 3:
                new_stop = self.entry_price
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.trailing_active = True
            
            if self.trailing_active and self.atr:
                trail_stop = self.highest_price - (self.atr * 2)
                if trail_stop > self.stop_loss:
                    self.stop_loss = trail_stop
        
    def close(self, exit_price, exit_date):
        self.exit_price = exit_price
        self.exit_date = exit_date
        if self.position_type == 'long':
            self.pnl = (exit_price - self.entry_price) * self.shares
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:
            self.pnl = (self.entry_price - exit_price) * self.shares
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100
        return self.pnl


class ImprovedTradingAccount:
    """Trading account with improved risk management"""
    
    def __init__(self, name, initial_balance, asset_class='stocks', max_drawdown_pct=40):
        self.name = name
        self.asset_class = asset_class
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.peak_equity = initial_balance
        self.max_drawdown = 0
        self.failed = False
        self.fail_reason = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.last_trade_date = None
        self.min_days_between_trades = 2
        
    def get_equity(self, current_prices):
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if position.position_type == 'long':
                    equity += position.shares * current_price
        return equity
    
    def update_equity_curve(self, date, current_prices):
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
    
    def can_trade(self, date):
        """Check if enough time has passed since last trade"""
        if self.last_trade_date is None:
            return True
        days_since = (date - self.last_trade_date).days
        return days_since >= self.min_days_between_trades
    
    def open_position(self, symbol, price, position_type, date, atr=None, risk_pct=1.5):
        """Open position with ATR-based stops"""
        if symbol in self.positions:
            return False
        
        if len(self.positions) >= 3:
            return False
        
        if not self.can_trade(date):
            return False
        
        if atr and not pd.isna(atr):
            stop_distance = atr * 2.0
            take_profit_distance = atr * 3.5
        else:
            stop_distance = price * 0.04
            take_profit_distance = price * 0.08
        
        risk_amount = self.balance * (risk_pct / 100)
        shares = risk_amount / stop_distance
        position_value = shares * price
        
        max_position = self.balance * 0.25
        if position_value > max_position:
            shares = max_position / price
            position_value = shares * price
        
        if position_value > self.balance * 0.95:
            shares = self.balance * 0.90 / price
            position_value = shares * price
        
        if position_value < 100:
            return False
        
        stop_loss = price - stop_distance if position_type == 'long' else price + stop_distance
        take_profit = price + take_profit_distance if position_type == 'long' else price - take_profit_distance
        
        position = Position(symbol, price, shares, position_type, date, stop_loss, take_profit, atr)
        self.positions[symbol] = position
        self.balance -= position_value
        self.last_trade_date = date
        return True
    
    def close_position(self, symbol, price, date, reason='signal'):
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        pnl = position.close(price, date)
        
        if position.position_type == 'long':
            self.balance += position.shares * price
        
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
    
    def check_stops(self, symbol, high, low, current_price, date):
        """Check stops with trailing stop update"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.update_trailing_stop(current_price)
        
        if position.position_type == 'long':
            if low <= position.stop_loss:
                exit_price = position.stop_loss
                reason = 'trailing_stop' if position.trailing_active else 'stop_loss'
                return self.close_position(symbol, exit_price, date, reason)
            if high >= position.take_profit:
                return self.close_position(symbol, position.take_profit, date, 'take_profit')
        return None
    
    def get_performance_summary(self):
        if not self.equity_curve:
            return {}
        
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity / self.initial_balance - 1) * 100
        
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if len(equity_df) > 1 and equity_df['returns'].std() > 0 else 0
        
        return {
            'account_name': self.name,
            'asset_class': self.asset_class,
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


class ImprovedBacktestEngine:
    """Backtest engine with improved strategy"""
    
    def fetch_data(self, symbol, start_date, end_date):
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                if len(df) > 0:
                    return df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return self.generate_simulated_data(symbol, start_date, end_date)
    
    def generate_simulated_data(self, symbol, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        base_prices = {
            'AAPL': 180, 'MSFT': 400, 'GOOGL': 140, 'AMZN': 180, 'NVDA': 500,
            'TSLA': 250, 'META': 500, 'AMD': 150, 'NFLX': 600, 'JPM': 200,
            'BTC-USD': 45000, 'ETH-USD': 2500, 'SOL-USD': 100, 'XRP-USD': 0.60, 'ADA-USD': 0.50,
            'DOGE-USD': 0.08, 'AVAX-USD': 35, 'DOT-USD': 7, 'MATIC-USD': 0.80, 'LINK-USD': 15,
            'EURUSD=X': 1.08, 'GBPUSD=X': 1.27, 'USDJPY=X': 150, 'AUDUSD=X': 0.65, 'USDCAD=X': 1.35
        }
        
        base_price = base_prices.get(symbol, 100)
        volatility = 0.012 if 'USD' in symbol and '=' in symbol else 0.025 if '-USD' in symbol else 0.018
        
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0005, volatility, len(dates))
        
        trend = np.linspace(0, 0.15, len(dates)) * (1 if np.random.random() > 0.4 else -1)
        returns = returns + trend / len(dates)
        
        prices = base_price * np.cumprod(1 + returns)
        
        daily_range = prices * np.random.uniform(0.008, 0.025, len(dates))
        opens = prices * (1 + np.random.uniform(-0.004, 0.004, len(dates)))
        highs = np.maximum(opens, prices) + daily_range * 0.5
        lows = np.minimum(opens, prices) - daily_range * 0.5
        volumes = np.random.randint(5000000, 100000000, len(dates))
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return df
    
    def run_backtest(self, account, symbols, start_date, end_date):
        print(f"\n{'='*60}")
        print(f"Running IMPROVED backtest for: {account.name}")
        print(f"Asset Class: {account.asset_class}")
        print(f"Symbols: {symbols}")
        print(f"{'='*60}")
        
        signal_generator = ImprovedSignalGenerator(account.asset_class)
        
        all_data = {}
        for symbol in symbols:
            df = self.fetch_data(symbol, start_date, end_date)
            if df is not None and len(df) > 60:
                df = signal_generator.calculate_indicators(df)
                all_data[symbol] = df
                print(f"Loaded {len(df)} days of data for {symbol}")
        
        if not all_data:
            print("No data available")
            return
        
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        trading_days = sorted(all_dates)
        
        for date in trading_days:
            if account.failed:
                break
            
            current_prices = {}
            
            for symbol, df in all_data.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                row = df.iloc[idx]
                current_prices[symbol] = row['Close']
                
                account.check_stops(symbol, row['High'], row['Low'], row['Close'], date)
                
                signal_result = signal_generator.generate_signal(df, idx)
                
                if signal_result['signal'] == 'BUY':
                    if symbol not in account.positions:
                        atr = signal_result.get('atr')
                        if account.open_position(symbol, row['Close'], 'long', date, atr):
                            print(f"  {date.date()}: BUY {symbol} @ ${row['Close']:.2f} (conf: {signal_result['confidence']:.2f})")
                
                elif signal_result['signal'] == 'SELL':
                    if symbol in account.positions:
                        pnl = account.close_position(symbol, row['Close'], date, 'sell_signal')
                        if pnl is not None:
                            print(f"  {date.date()}: SELL {symbol} @ ${row['Close']:.2f} (PnL: ${pnl:.2f})")
            
            if current_prices:
                if not account.update_equity_curve(date, current_prices):
                    print(f"\n*** ACCOUNT FAILED: {account.fail_reason} ***")
                    break
        
        for symbol in list(account.positions.keys()):
            if symbol in all_data and len(all_data[symbol]) > 0:
                final_price = all_data[symbol]['Close'].iloc[-1]
                account.close_position(symbol, final_price, trading_days[-1], 'end_of_period')
        
        return account.get_performance_summary()


def run_improved_experiment():
    """Run the improved paper trading experiment"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'JPM']
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD']
    forex_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    
    stock_account = ImprovedTradingAccount("Stocks Portfolio V2", 10000, 'stocks')
    crypto_account = ImprovedTradingAccount("Crypto Portfolio V2", 10000, 'crypto')
    forex_account = ImprovedTradingAccount("Forex Portfolio V2", 10000, 'forex')
    
    engine = ImprovedBacktestEngine()
    
    results = {}
    results['stocks'] = engine.run_backtest(stock_account, stock_symbols, start_date, end_date)
    results['crypto'] = engine.run_backtest(crypto_account, crypto_symbols, start_date, end_date)
    results['forex'] = engine.run_backtest(forex_account, forex_symbols, start_date, end_date)
    
    print("\n" + "="*80)
    print("IMPROVED PAPER TRADING EXPERIMENT RESULTS (V2)")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: $10,000 per portfolio ($30,000 total)")
    print("\nIMPROVEMENTS APPLIED:")
    print("  - Trend filter (only trade with SMA 50 direction)")
    print("  - Higher confidence thresholds (70-75%)")
    print("  - Dynamic ATR-based stop losses")
    print("  - Trailing stops to lock profits")
    print("  - Volume confirmation")
    print("  - Reduced trading frequency")
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
    print("COMBINED RESULTS (V2)")
    print(f"{'='*80}")
    print(f"  Total Final Equity:  ${total_final:,.2f}")
    print(f"  Total Return:        {((total_final / 30000) - 1) * 100:+.2f}%")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Overall Win Rate:    {(total_wins / total_trades * 100) if total_trades > 0 else 0:.1f}%")
    
    print(f"\n{'='*80}")
    print("COMPARISON: V1 vs V2")
    print(f"{'='*80}")
    print("  V1 Total Return:  -13.93%")
    print(f"  V2 Total Return:  {((total_final / 30000) - 1) * 100:+.2f}%")
    print("  V1 Win Rate:      31.2%")
    print(f"  V2 Win Rate:      {(total_wins / total_trades * 100) if total_trades > 0 else 0:.1f}%")
    print("  V1 Trades:        202")
    print(f"  V2 Trades:        {total_trades}")
    
    results_file = 'backtesting/experiment_results_v2.json'
    with open(results_file, 'w') as f:
        json.dump({
            'version': 2,
            'run_date': datetime.now().isoformat(),
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'improvements': [
                'Trend filter (SMA 50)',
                'Higher confidence thresholds',
                'Dynamic ATR-based stops',
                'Trailing stops',
                'Volume confirmation',
                'Reduced trading frequency'
            ],
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_improved_experiment()
