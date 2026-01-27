"""
Paper Trading Experiment V3 - BALANCED ADVANCED STRATEGY
=========================================================
Balances between V2's signal generation with improved:
1. Partial profit taking (scale out 50% at first target)
2. Multi-level trailing stops
3. Better risk/reward ratios
4. Asset-specific position limits
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


class BalancedSignalGenerator:
    """Balanced signal generator - keeps V2 entry logic with refinements"""
    
    def __init__(self, asset_class='stocks'):
        self.asset_class = asset_class
        self.confidence_thresholds = {
            'stocks': 0.68,
            'crypto': 0.72,
            'forex': 0.68
        }
        
    def calculate_indicators(self, df):
        df = df.copy()
        
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
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
        
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        df['momentum_pct'] = df['Close'].pct_change(periods=10) * 100
        
        return df
    
    def generate_signal(self, df, idx):
        if idx < 55:
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
                reasons.append("SMA crossover bullish")
            elif row['sma_10'] < row['sma_20'] and prev_row['sma_10'] >= prev_row['sma_20']:
                sell_signals += 3
                reasons.append("SMA crossover bearish")
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
            elif row['macd'] > row['macd_signal']:
                buy_signals += 1
            else:
                sell_signals += 1
        
        rsi_buy = 28 if self.asset_class == 'crypto' else 30
        rsi_sell = 72 if self.asset_class == 'crypto' else 70
        
        if pd.notna(row['rsi']):
            if row['rsi'] < rsi_buy:
                buy_signals += 3
                reasons.append(f"RSI oversold ({row['rsi']:.1f})")
            elif row['rsi'] > rsi_sell:
                sell_signals += 3
                reasons.append(f"RSI overbought ({row['rsi']:.1f})")
            elif row['rsi'] < 38:
                buy_signals += 1
            elif row['rsi'] > 62:
                sell_signals += 1
        
        if pd.notna(row['bb_lower']) and pd.notna(row['bb_upper']):
            bb_pct = (row['Close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if bb_pct < 0.1:
                buy_signals += 2
                reasons.append("Near lower BB")
            elif bb_pct > 0.9:
                sell_signals += 2
                reasons.append("Near upper BB")
            elif bb_pct < 0.25:
                buy_signals += 1
            elif bb_pct > 0.75:
                sell_signals += 1
        
        if pd.notna(row['momentum_pct']):
            if row['momentum_pct'] > 6:
                buy_signals += 1
            elif row['momentum_pct'] < -6:
                sell_signals += 1
        
        total = buy_signals + sell_signals
        if total == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No signals', 'atr': row.get('atr')}
        
        buy_ratio = buy_signals / total
        sell_ratio = sell_signals / total
        
        threshold = self.confidence_thresholds.get(self.asset_class, 0.68)
        
        if buy_ratio > 0.55:
            if not in_uptrend:
                return {'signal': 'HOLD', 'confidence': buy_ratio * 0.5,
                        'reason': 'Buy rejected - against trend', 'atr': row.get('atr')}
            
            confidence = min(0.95, buy_ratio)
            if volume_confirmed:
                confidence = min(0.98, confidence * 1.1)
                reasons.append("Volume confirmed")
            
            if confidence >= threshold:
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'reason': '; '.join(reasons) if reasons else 'Bullish',
                    'atr': row.get('atr')
                }
        
        elif sell_ratio > 0.55:
            if not in_downtrend:
                return {'signal': 'HOLD', 'confidence': sell_ratio * 0.5,
                        'reason': 'Sell rejected - against trend', 'atr': row.get('atr')}
            
            confidence = min(0.95, sell_ratio)
            if volume_confirmed:
                confidence = min(0.98, confidence * 1.1)
            
            if confidence >= threshold:
                return {
                    'signal': 'SELL',
                    'confidence': confidence,
                    'reason': '; '.join(reasons) if reasons else 'Bearish',
                    'atr': row.get('atr')
                }
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Mixed signals', 'atr': row.get('atr')}


class ScaledPosition:
    """Position with scaled exits"""
    def __init__(self, symbol, entry_price, shares, position_type, entry_date, 
                 stop_loss, take_profit_1, take_profit_2, atr=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.initial_shares = shares
        self.shares = shares
        self.position_type = position_type
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.initial_stop = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.atr = atr
        self.highest_price = entry_price
        self.trailing_active = False
        self.partial_done = False
        self.exit_price = None
        self.exit_date = None
        self.pnl = 0
        self.pnl_pct = 0
        self.partial_pnl = 0
        
    def update_trailing(self, current_price):
        if self.position_type == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            profit_pct = (current_price / self.entry_price - 1) * 100
            
            if profit_pct >= 6:
                new_stop = self.entry_price * 1.03
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.trailing_active = True
            elif profit_pct >= 4:
                new_stop = self.entry_price * 1.015
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.trailing_active = True
            elif profit_pct >= 2.5:
                new_stop = self.entry_price
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
                    self.trailing_active = True
            
            if self.trailing_active and self.atr and profit_pct >= 4:
                trail = self.highest_price - (self.atr * 1.8)
                if trail > self.stop_loss:
                    self.stop_loss = trail
    
    def check_partial(self, high):
        if self.partial_done:
            return None
        
        if self.position_type == 'long' and high >= self.take_profit_1:
            exit_shares = self.shares * 0.5
            self.partial_pnl = (self.take_profit_1 - self.entry_price) * exit_shares
            self.shares -= exit_shares
            self.partial_done = True
            self.stop_loss = max(self.stop_loss, self.entry_price * 1.005)
            return self.partial_pnl
        return None
        
    def close(self, exit_price, exit_date):
        self.exit_price = exit_price
        self.exit_date = exit_date
        if self.position_type == 'long':
            self.pnl = (exit_price - self.entry_price) * self.shares + self.partial_pnl
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        return self.pnl


class ScaledTradingAccount:
    """Account with scaled exits"""
    
    def __init__(self, name, initial_balance, asset_class='stocks', max_dd=40):
        self.name = name
        self.asset_class = asset_class
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_dd = max_dd
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
        
        self.cfg = {
            'stocks': {'max_pos': 3, 'min_days': 2, 'risk': 1.5, 'max_pct': 0.25},
            'crypto': {'max_pos': 2, 'min_days': 2, 'risk': 1.2, 'max_pct': 0.30},
            'forex': {'max_pos': 2, 'min_days': 2, 'risk': 1.5, 'max_pct': 0.30}
        }
        
    def get_equity(self, prices):
        equity = self.balance
        for sym, pos in self.positions.items():
            if sym in prices:
                equity += pos.shares * prices[sym]
        return equity
    
    def update_equity(self, date, prices):
        equity = self.get_equity(prices)
        self.equity_curve.append({'date': date, 'equity': equity})
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        dd = (self.peak_equity - equity) / self.peak_equity * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        
        if dd >= self.max_dd:
            self.failed = True
            self.fail_reason = f"Max drawdown: {dd:.2f}%"
            return False
        return True
    
    def can_trade(self, date):
        if self.last_trade_date is None:
            return True
        min_days = self.cfg[self.asset_class]['min_days']
        return (date - self.last_trade_date).days >= min_days
    
    def open_position(self, symbol, price, pos_type, date, atr=None):
        if symbol in self.positions:
            return False
        
        c = self.cfg[self.asset_class]
        if len(self.positions) >= c['max_pos']:
            return False
        
        if not self.can_trade(date):
            return False
        
        if atr and not pd.isna(atr):
            stop_dist = atr * 2.0
            tp1_dist = atr * 2.5
            tp2_dist = atr * 4.0
        else:
            stop_dist = price * 0.04
            tp1_dist = price * 0.05
            tp2_dist = price * 0.08
        
        risk_amt = self.balance * (c['risk'] / 100)
        shares = risk_amt / stop_dist
        pos_val = shares * price
        
        max_pos = self.balance * c['max_pct']
        if pos_val > max_pos:
            shares = max_pos / price
            pos_val = shares * price
        
        if pos_val > self.balance * 0.95:
            shares = self.balance * 0.90 / price
            pos_val = shares * price
        
        if pos_val < 100:
            return False
        
        stop = price - stop_dist
        tp1 = price + tp1_dist
        tp2 = price + tp2_dist
        
        self.positions[symbol] = ScaledPosition(symbol, price, shares, pos_type, date, stop, tp1, tp2, atr)
        self.balance -= pos_val
        self.last_trade_date = date
        return True
    
    def close_position(self, symbol, price, date, reason='signal'):
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pnl = pos.close(price, date)
        
        self.balance += pos.shares * price
        
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        pos.close_reason = reason
        self.closed_trades.append(pos)
        del self.positions[symbol]
        return pnl
    
    def check_exits(self, symbol, high, low, current, date):
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        partial = pos.check_partial(high)
        if partial:
            print(f"    Partial: {symbol} +${partial:.2f}")
        
        pos.update_trailing(current)
        
        if pos.position_type == 'long':
            if low <= pos.stop_loss:
                reason = 'trailing' if pos.trailing_active else 'stop'
                return self.close_position(symbol, pos.stop_loss, date, reason)
            if high >= pos.take_profit_2:
                return self.close_position(symbol, pos.take_profit_2, date, 'target')
        return None
    
    def get_summary(self):
        if not self.equity_curve:
            return {}
        
        final = self.equity_curve[-1]['equity']
        ret = (final / self.initial_balance - 1) * 100
        
        wr = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        avg_w = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_l = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        pf = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df['ret'] = eq_df['equity'].pct_change()
        std = eq_df['ret'].std()
        sharpe = eq_df['ret'].mean() / std * np.sqrt(252) if std > 0 else 0
        
        return {
            'account_name': self.name,
            'initial_balance': self.initial_balance,
            'final_equity': round(final, 2),
            'total_return_pct': round(ret, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': round(wr, 2),
            'avg_win': round(avg_w, 2),
            'avg_loss': round(avg_l, 2),
            'profit_factor': round(pf, 2) if pf != float('inf') else 'N/A',
            'max_drawdown_pct': round(self.max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2) if not np.isnan(sharpe) else 0,
            'failed': self.failed,
            'fail_reason': self.fail_reason
        }


class V3BacktestEngine:
    
    def fetch_data(self, symbol, start, end):
        if YFINANCE_AVAILABLE:
            try:
                df = yf.Ticker(symbol).history(start=start, end=end, interval='1d')
                if len(df) > 0:
                    return df
            except Exception as e:
                print(f"Error {symbol}: {e}")
        return self._sim_data(symbol, start, end)
    
    def _sim_data(self, symbol, start, end):
        dates = pd.date_range(start=start, end=end, freq='B')
        
        base = {'AAPL': 180, 'MSFT': 400, 'GOOGL': 140, 'AMZN': 180, 'NVDA': 500,
                'TSLA': 250, 'META': 500, 'AMD': 150, 'NFLX': 600, 'JPM': 200,
                'BTC-USD': 45000, 'ETH-USD': 2500, 'SOL-USD': 100, 'XRP-USD': 0.60, 'ADA-USD': 0.50,
                'EURUSD=X': 1.08, 'GBPUSD=X': 1.27, 'USDJPY=X': 150}.get(symbol, 100)
        
        vol = 0.010 if '=' in symbol else 0.022 if '-USD' in symbol else 0.016
        
        np.random.seed(hash(symbol) % 2**32)
        rets = np.random.normal(0.0006, vol, len(dates))
        trend = np.linspace(0, 0.15, len(dates)) * (1 if np.random.random() > 0.4 else -1)
        rets += trend / len(dates)
        
        prices = base * np.cumprod(1 + rets)
        rng = prices * np.random.uniform(0.008, 0.022, len(dates))
        
        return pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.003, 0.003, len(dates))),
            'High': np.maximum(prices, prices) + rng * 0.5,
            'Low': np.minimum(prices, prices) - rng * 0.5,
            'Close': prices,
            'Volume': np.random.randint(5000000, 100000000, len(dates))
        }, index=dates)
    
    def run(self, account, symbols, start, end):
        print(f"\n{'='*60}")
        print(f"V3 BALANCED: {account.name}")
        print(f"{'='*60}")
        
        sig_gen = BalancedSignalGenerator(account.asset_class)
        
        data = {}
        for sym in symbols:
            df = self.fetch_data(sym, start, end)
            if df is not None and len(df) > 55:
                df = sig_gen.calculate_indicators(df)
                data[sym] = df
                print(f"Loaded {len(df)} days: {sym}")
        
        if not data:
            return None
        
        all_dates = sorted(set(d for df in data.values() for d in df.index.tolist()))
        
        for date in all_dates:
            if account.failed:
                break
            
            prices = {}
            
            for sym, df in data.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                row = df.iloc[idx]
                prices[sym] = row['Close']
                
                account.check_exits(sym, row['High'], row['Low'], row['Close'], date)
                
                sig = sig_gen.generate_signal(df, idx)
                
                if sig['signal'] == 'BUY' and sym not in account.positions:
                    if account.open_position(sym, row['Close'], 'long', date, sig.get('atr')):
                        print(f"  {date.date()}: BUY {sym} @ ${row['Close']:.2f} (conf: {sig['confidence']:.2f})")
                
                elif sig['signal'] == 'SELL' and sym in account.positions:
                    pnl = account.close_position(sym, row['Close'], date, 'signal')
                    if pnl:
                        print(f"  {date.date()}: SELL {sym} @ ${row['Close']:.2f} (${pnl:+.2f})")
            
            if prices:
                account.update_equity(date, prices)
        
        for sym in list(account.positions.keys()):
            if sym in data:
                account.close_position(sym, data[sym]['Close'].iloc[-1], all_dates[-1], 'end')
        
        return account.get_summary()


def run_v3():
    end = datetime.now()
    start = end - timedelta(days=180)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'JPM']
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD']
    forex = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    
    stock_acc = ScaledTradingAccount("Stocks V3", 10000, 'stocks')
    crypto_acc = ScaledTradingAccount("Crypto V3", 10000, 'crypto')
    forex_acc = ScaledTradingAccount("Forex V3", 10000, 'forex')
    
    engine = V3BacktestEngine()
    
    results = {
        'stocks': engine.run(stock_acc, stocks, start, end),
        'crypto': engine.run(crypto_acc, crypto, start, end),
        'forex': engine.run(forex_acc, forex, start, end)
    }
    
    print("\n" + "="*80)
    print("V3 BALANCED RESULTS")
    print("="*80)
    print(f"Period: {start.date()} to {end.date()}")
    print("\nV3 FEATURES:")
    print("  - Trend filter + balanced thresholds")
    print("  - 50% partial profit at first target")
    print("  - Multi-level trailing stops")
    print("  - Move to breakeven at +2.5%")
    print("="*80)
    
    total_final = 0
    total_trades = 0
    total_wins = 0
    
    for name, s in results.items():
        if s:
            print(f"\n{'-'*40}")
            print(f"{s['account_name'].upper()}")
            print(f"{'-'*40}")
            print(f"  Final Equity:     ${s['final_equity']:,.2f}")
            print(f"  Total Return:     {s['total_return_pct']:+.2f}%")
            print(f"  Total Trades:     {s['total_trades']}")
            print(f"  Win Rate:         {s['win_rate_pct']:.1f}%")
            print(f"  Profit Factor:    {s['profit_factor']}")
            print(f"  Max Drawdown:     {s['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe Ratio:     {s['sharpe_ratio']:.2f}")
            
            total_final += s['final_equity']
            total_trades += s['total_trades']
            total_wins += s['winning_trades']
    
    print(f"\n{'='*80}")
    print("COMBINED V3")
    print(f"{'='*80}")
    print(f"  Total Final:  ${total_final:,.2f}")
    print(f"  Total Return: {((total_final / 30000) - 1) * 100:+.2f}%")
    print(f"  Total Trades: {total_trades}")
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  Win Rate:     {wr:.1f}%")
    
    print(f"\n{'='*80}")
    print("VERSION COMPARISON")
    print(f"{'='*80}")
    print("  V1:  -13.93% | 202 trades | 31.2% win rate")
    print("  V2:   -1.82% |  32 trades | 31.2% win rate")
    print(f"  V3:  {((total_final / 30000) - 1) * 100:+.2f}% | {total_trades:3} trades | {wr:.1f}% win rate")
    
    with open('backtesting/experiment_results_v3.json', 'w') as f:
        json.dump({
            'version': 3,
            'run_date': datetime.now().isoformat(),
            'results': results,
            'comparison': {
                'v1': {'return': -13.93, 'trades': 202, 'win_rate': 31.2},
                'v2': {'return': -1.82, 'trades': 32, 'win_rate': 31.2},
                'v3': {'return': round((total_final / 30000 - 1) * 100, 2), 'trades': total_trades, 'win_rate': round(wr, 1)}
            }
        }, f, indent=2, default=str)
    
    print(f"\nSaved: backtesting/experiment_results_v3.json")
    return results


if __name__ == "__main__":
    run_v3()
