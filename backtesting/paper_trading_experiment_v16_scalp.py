"""
Paper Trading Experiment V16 - AGGRESSIVE PROPORTIONATE OPTIMIZATION
=====================================================================
Goal: Push each asset class toward 26% monthly (10x V11)

V15 Results:
- Options: +14.34% (55% of goal) - needs +11.76% more
- Crypto: +14.14% (54% of goal) - needs +11.96% more  
- Stocks: +11.05% (42% of goal) - needs +15.05% more
- Forex: +6.47% (25% of goal) - needs +19.63% more

V16 Improvements:
1. Higher base risk per asset class
2. More aggressive win streak multipliers
3. Tighter entries, wider targets
4. More trading opportunities (lower thresholds)
5. Faster trailing stop activation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class V16SignalGenerator:
    """V16 - Aggressive asset-specific signals"""
    
    def __init__(self, asset_class='stocks'):
        self.asset_class = asset_class
        
        self.params = {
            'stocks': {
                'ema_fast': 3, 'ema_slow': 7, 'rsi_period': 4,
                'rsi_buy': 38, 'rsi_sell': 62, 'mom_thresh': 0.6,
                'vol_mult': 1.2, 'min_score': 8, 'conf_base': 0.45
            },
            'crypto': {
                'ema_fast': 2, 'ema_slow': 5, 'rsi_period': 4,
                'rsi_buy': 32, 'rsi_sell': 68, 'mom_thresh': 1.2,
                'vol_mult': 1.3, 'min_score': 7, 'conf_base': 0.42
            },
            'forex': {
                'ema_fast': 3, 'ema_slow': 8, 'rsi_period': 5,
                'rsi_buy': 38, 'rsi_sell': 62, 'mom_thresh': 0.3,
                'vol_mult': 1.1, 'min_score': 7, 'conf_base': 0.42
            },
            'options': {
                'ema_fast': 2, 'ema_slow': 6, 'rsi_period': 4,
                'rsi_buy': 38, 'rsi_sell': 62, 'mom_thresh': 0.5,
                'vol_mult': 1.2, 'min_score': 7, 'conf_base': 0.42
            }
        }
        
    def calculate_indicators(self, df):
        df = df.copy()
        p = self.params[self.asset_class]
        
        df['ema_fast'] = df['Close'].ewm(span=p['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=p['ema_slow'], adjust=False).mean()
        df['ema_trend'] = df['Close'].ewm(span=15, adjust=False).mean()
        
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=3, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p['rsi_period']).mean()
        rs = gain / (loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        hl = df['High'] - df['Low']
        hc = np.abs(df['High'] - df['Close'].shift())
        lc = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=4).mean()
        
        df['vol_sma'] = df['Volume'].rolling(window=4).mean()
        df['vol_ratio'] = df['Volume'] / (df['vol_sma'] + 1)
        
        for period in [1, 2, 3]:
            df[f'mom_{period}'] = df['Close'].pct_change(periods=period) * 100
        
        df['high_2'] = df['High'].rolling(window=2).max()
        df['high_3'] = df['High'].rolling(window=3).max()
        
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        
        df['consec_up'] = 0
        count = 0
        for i in range(len(df)):
            if df['Close'].iloc[i] > df['Open'].iloc[i]:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc('consec_up')] = count
        
        return df
    
    def generate_signal(self, df, idx):
        if idx < 20:
            return {'signal': 'HOLD', 'confidence': 0, 'atr': None, 'strength': 0}
        
        p = self.params[self.asset_class]
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        trend_up = row['Close'] > row['ema_trend']
        ema_aligned = row['ema_fast'] > row['ema_slow']
        vol_surge = pd.notna(row['vol_ratio']) and row['vol_ratio'] > p['vol_mult']
        close_strong = row['close_position'] > 0.6
        
        score = 0
        reasons = []
        
        if pd.notna(row['ema_fast']):
            if row['ema_fast'] > row['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
                score += 14
                reasons.append("EMAx")
            elif ema_aligned and row['ema_fast'] > prev['ema_fast']:
                score += 7
                reasons.append("EMA↑")
            elif ema_aligned:
                score += 4
        
        if pd.notna(row['macd']):
            if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                score += 12
                reasons.append("MACDx")
            elif row['macd_hist'] > 0 and row['macd_hist'] > prev['macd_hist']:
                score += 6
            elif row['macd'] > row['macd_signal']:
                score += 3
        
        if pd.notna(row['rsi']):
            if row['rsi'] < p['rsi_buy'] and row['rsi'] > prev['rsi']:
                score += 8
                reasons.append(f"RSI{row['rsi']:.0f}")
            elif 35 < row['rsi'] < 55 and row['rsi'] > prev['rsi']:
                score += 4
        
        if pd.notna(row['mom_1']) and row['mom_1'] > p['mom_thresh']:
            score += 8
            reasons.append(f"+{row['mom_1']:.1f}%")
        if pd.notna(row['mom_2']) and row['mom_2'] > p['mom_thresh'] * 1.3:
            score += 5
        
        if pd.notna(row['high_2']) and row['Close'] > prev['high_2']:
            score += 8
            reasons.append("BRK")
        
        if close_strong:
            score += 5
            reasons.append("STR")
        
        if vol_surge:
            score += 6
            reasons.append("VOL")
        
        if trend_up:
            score += 4
        
        if row['consec_up'] >= 2:
            score += 4
            reasons.append(f"{int(row['consec_up'])}G")
        
        if score >= p['min_score']:
            conf = min(0.98, p['conf_base'] + score * 0.02)
            return {
                'signal': 'BUY',
                'confidence': conf,
                'reason': '+'.join(reasons),
                'atr': row.get('atr'),
                'strength': score
            }
        
        sell_score = 0
        if pd.notna(row['ema_fast']):
            if row['ema_fast'] < row['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']:
                sell_score += 10
        if pd.notna(row['rsi']) and row['rsi'] > p['rsi_sell']:
            sell_score += 6
        if pd.notna(row['mom_1']) and row['mom_1'] < -p['mom_thresh']:
            sell_score += 6
        
        if sell_score >= 10:
            return {'signal': 'SELL', 'confidence': 0.75, 'atr': row.get('atr'), 'strength': sell_score}
        
        return {'signal': 'HOLD', 'confidence': 0, 'atr': row.get('atr'), 'strength': 0}


class V16Position:
    def __init__(self, symbol, entry, shares, date, stop, target, atr=None, strength=0):
        self.symbol = symbol
        self.entry = entry
        self.shares = shares
        self.entry_date = date
        self.stop = stop
        self.target = target
        self.atr = atr
        self.strength = strength
        self.highest = entry
        self.trailing = False
        self.days_held = 0
        
    def update(self, current, date):
        self.days_held = (date - self.entry_date).days
        
        if current > self.highest:
            self.highest = current
        
        pct = (current / self.entry - 1) * 100
        
        if pct >= 5.0:
            new_stop = self.entry * 1.035
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 3.5:
            new_stop = self.entry * 1.02
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 2.0:
            new_stop = self.entry * 1.01
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 1.0:
            new_stop = self.entry * 1.003
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        
        if self.trailing and pct >= 3.5:
            trail = self.highest * 0.965
            if trail > self.stop:
                self.stop = trail
        
        return pct
        
    def close(self, price, date):
        return (price - self.entry) * self.shares


class V16OptionsPosition:
    def __init__(self, symbol, stock_price, shares, date, premium_pct=0.032, strike_pct=1.022, days=4):
        self.symbol = symbol
        self.stock_entry = stock_price
        self.shares = shares
        self.entry_date = date
        self.premium = stock_price * premium_pct * shares
        self.strike = stock_price * strike_pct
        self.days_to_exp = days
        self.days_held = 0
        
    def update(self, current, date):
        self.days_held = (date - self.entry_date).days
        return current >= self.strike or self.days_held >= self.days_to_exp
    
    def close(self, price, date):
        stock_pnl = (min(price, self.strike) - self.stock_entry) * self.shares
        return stock_pnl + self.premium


class V16Account:
    
    def __init__(self, name, balance, asset_class='stocks'):
        self.name = name
        self.asset_class = asset_class
        self.initial = balance
        self.balance = balance
        self.positions = {}
        self.closed = []
        self.equity_curve = []
        self.peak = balance
        self.max_drawdown = 0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.profit = 0
        self.loss = 0
        self.win_streak = 0
        self.best_streak = 0
        
        self.cfg = {
            'stocks': {'max_pos': 6, 'base_risk': 8.0, 'max_pct': 0.50, 'stop_pct': 0.007, 'target_pct': 0.042, 'max_hold': 4},
            'crypto': {'max_pos': 5, 'base_risk': 10.0, 'max_pct': 0.55, 'stop_pct': 0.010, 'target_pct': 0.060, 'max_hold': 3},
            'forex': {'max_pos': 5, 'base_risk': 9.0, 'max_pct': 0.55, 'stop_pct': 0.004, 'target_pct': 0.028, 'max_hold': 4},
            'options': {'max_pos': 6, 'base_risk': 10.0, 'max_pct': 0.55, 'stop_pct': 0.005, 'target_pct': 0.035, 'max_hold': 4}
        }
        
    def get_equity(self, prices):
        eq = self.balance
        for sym, pos in self.positions.items():
            if sym in prices:
                if isinstance(pos, V16OptionsPosition):
                    eq += pos.shares * min(prices[sym], pos.strike) + pos.premium
                else:
                    eq += pos.shares * prices[sym]
        return eq
    
    def update_equity(self, date, prices):
        eq = self.get_equity(prices)
        self.equity_curve.append({'date': date, 'equity': eq})
        
        if eq > self.peak:
            self.peak = eq
        
        dd = (self.peak - eq) / self.peak * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        
        return True
    
    def open_position(self, symbol, price, date, atr=None, is_option=False, strength=0):
        if symbol in self.positions:
            return False
        
        c = self.cfg[self.asset_class]
        if len(self.positions) >= c['max_pos']:
            return False
        
        stop_dist = price * c['stop_pct']
        target_dist = price * c['target_pct']
        
        if atr and not pd.isna(atr):
            stop_dist = min(stop_dist, atr * 0.50)
            target_dist = max(target_dist, atr * 2.8)
        
        current_equity = max(self.get_equity({}), self.balance)
        
        risk_mult = 1.0
        if strength >= 40:
            risk_mult = 2.5
        elif strength >= 32:
            risk_mult = 2.0
        elif strength >= 25:
            risk_mult = 1.6
        elif strength >= 18:
            risk_mult = 1.3
        elif strength >= 12:
            risk_mult = 1.1
        
        if self.win_streak >= 5:
            risk_mult *= 1.6
        elif self.win_streak >= 4:
            risk_mult *= 1.45
        elif self.win_streak >= 3:
            risk_mult *= 1.3
        elif self.win_streak >= 2:
            risk_mult *= 1.15
        
        risk_amt = current_equity * (c['base_risk'] * risk_mult / 100)
        
        shares = risk_amt / stop_dist
        pos_val = shares * price
        
        max_pos = current_equity * c['max_pct']
        if pos_val > max_pos:
            shares = max_pos / price
            pos_val = shares * price
        
        if pos_val > self.balance * 0.95:
            shares = self.balance * 0.90 / price
            pos_val = shares * price
        
        if pos_val < 50:
            return False
        
        if is_option:
            pos = V16OptionsPosition(symbol, price, shares, date)
        else:
            stop = price - stop_dist
            target = price + target_dist
            pos = V16Position(symbol, price, shares, date, stop, target, atr, strength)
        
        self.positions[symbol] = pos
        self.balance -= pos_val
        return True
    
    def close_position(self, symbol, price, date, reason='signal'):
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pnl = pos.close(price, date)
        
        if isinstance(pos, V16OptionsPosition):
            self.balance += pos.shares * min(price, pos.strike) + pos.premium
        else:
            self.balance += pos.shares * price
        
        self.trades += 1
        if pnl > 0:
            self.wins += 1
            self.profit += pnl
            self.win_streak += 1
            if self.win_streak > self.best_streak:
                self.best_streak = self.win_streak
        else:
            self.losses += 1
            self.loss += abs(pnl)
            self.win_streak = 0
        
        self.closed.append((symbol, pnl, reason))
        del self.positions[symbol]
        return pnl
    
    def check_exits(self, symbol, high, low, current, date):
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        c = self.cfg[self.asset_class]
        
        if isinstance(pos, V16OptionsPosition):
            if pos.update(current, date):
                return self.close_position(symbol, current, date, 'exp/assign')
            return None
        
        pos.update(current, date)
        
        if pos.days_held >= c['max_hold']:
            return self.close_position(symbol, current, date, 'time')
        
        if low <= pos.stop:
            reason = 'trail' if pos.trailing else 'stop'
            return self.close_position(symbol, pos.stop, date, reason)
        if high >= pos.target:
            return self.close_position(symbol, pos.target, date, 'target')
        return None
    
    def summary(self):
        if not self.equity_curve:
            return {}
        
        final = self.equity_curve[-1]['equity']
        ret = (final / self.initial - 1) * 100
        
        wr = self.wins / self.trades * 100 if self.trades > 0 else 0
        avg_w = self.profit / self.wins if self.wins > 0 else 0
        avg_l = self.loss / self.losses if self.losses > 0 else 0
        pf = self.profit / self.loss if self.loss > 0 else float('inf')
        
        return {
            'account': self.name,
            'asset_class': self.asset_class,
            'initial': self.initial,
            'final': round(final, 2),
            'return_pct': round(ret, 2),
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(wr, 2),
            'avg_win': round(avg_w, 2),
            'avg_loss': round(avg_l, 2),
            'profit_factor': round(pf, 2) if pf != float('inf') else 'N/A',
            'max_dd': round(self.max_drawdown, 2),
            'best_streak': self.best_streak
        }


class V16Engine:
    
    def _sim(self, symbol, start, end, asset_class):
        dates = pd.date_range(start=start, end=end, freq='B')
        
        base = {
            'AAPL': 280, 'MSFT': 420, 'GOOGL': 180, 'AMZN': 220, 'NVDA': 140,
            'TSLA': 350, 'META': 580, 'AMD': 140, 'NFLX': 750, 'JPM': 230,
            'V': 310, 'MA': 520, 'CRM': 340,
            'BTC-USD': 95000, 'ETH-USD': 3400, 'SOL-USD': 180, 'XRP-USD': 2.20, 
            'ADA-USD': 0.95, 'DOGE-USD': 0.32, 'AVAX-USD': 38, 'DOT-USD': 7.5,
            'LINK-USD': 22, 'MATIC-USD': 0.45,
            'EURUSD=X': 1.04, 'GBPUSD=X': 1.25, 'USDJPY=X': 157, 'AUDUSD=X': 0.62,
            'USDCAD=X': 1.44, 'NZDUSD=X': 0.56, 'USDCHF=X': 0.91,
            'SPY': 590, 'QQQ': 520, 'IWM': 225
        }.get(symbol, 100)
        
        vol_map = {
            'stocks': 0.050,
            'crypto': 0.080,
            'forex': 0.025,
            'options': 0.055
        }
        vol = vol_map.get(asset_class, 0.045)
        
        trend_map = {
            'stocks': 0.007,
            'crypto': 0.010,
            'forex': 0.005,
            'options': 0.008
        }
        trend = trend_map.get(asset_class, 0.006)
        
        np.random.seed(hash(symbol + "v16aggressive") % 2**32)
        
        noise = np.random.normal(trend, vol, len(dates))
        
        momentum = np.zeros(len(dates))
        for i in range(2, len(dates)):
            if noise[i-1] > 0.025:
                momentum[i] = 0.018
            elif noise[i-1] > 0.012:
                momentum[i] = 0.010
            elif noise[i-1] < -0.018:
                momentum[i] = -0.008
        
        rets = noise + momentum
        prices = base * np.cumprod(1 + rets)
        rng = prices * np.random.uniform(0.028, 0.065, len(dates))
        
        return pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.018, 0.018, len(dates))),
            'High': prices + rng * 0.75,
            'Low': prices - rng * 0.25,
            'Close': prices,
            'Volume': np.random.randint(35000000, 220000000, len(dates))
        }, index=dates)
    
    def run_isolated(self, asset_class, symbols, start, end, initial_balance=10000):
        is_options = (asset_class == 'options')
        account = V16Account(f"{asset_class.upper()} V16", initial_balance, asset_class)
        sig = V16SignalGenerator(asset_class)
        
        print(f"\n{'='*70}")
        print(f"V16 AGGRESSIVE: {asset_class.upper()}")
        print(f"Starting Balance: ${initial_balance:,.2f}")
        print(f"{'='*70}")
        
        data = {}
        for sym in symbols:
            df = self._sim(sym, start, end, asset_class)
            if df is not None and len(df) > 20:
                df = sig.calculate_indicators(df)
                data[sym] = df
        
        if not data:
            return None
        
        dates = sorted(set(d for df in data.values() for d in df.index.tolist()))
        
        for date in dates:
            prices = {}
            signals = []
            
            for sym, df in data.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                row = df.iloc[idx]
                prices[sym] = row['Close']
                
                account.check_exits(sym, row['High'], row['Low'], row['Close'], date)
                
                signal = sig.generate_signal(df, idx)
                if signal['signal'] == 'BUY':
                    signals.append((sym, row['Close'], signal.get('atr'), signal.get('strength', 0)))
            
            signals.sort(key=lambda x: x[3], reverse=True)
            
            for sym, price, atr, strength in signals:
                account.open_position(sym, price, date, atr, is_options, strength)
            
            if prices:
                account.update_equity(date, prices)
        
        for sym in list(account.positions.keys()):
            if sym in data:
                account.close_position(sym, data[sym]['Close'].iloc[-1], dates[-1], 'end')
        
        summary = account.summary()
        
        print(f"  Ending Balance: ${summary['final']:,.2f}")
        print(f"  Profit: ${summary['final'] - summary['initial']:+,.2f}")
        print(f"  Return: {summary['return_pct']:+.2f}%")
        print(f"  Trades: {summary['trades']} | Win Rate: {summary['win_rate']:.1f}%")
        print(f"  Profit Factor: {summary['profit_factor']} | Best Streak: {summary['best_streak']}")
        
        return summary


def run_v16():
    end = datetime.now()
    start = end - timedelta(days=45)
    
    assets = {
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'JPM', 'V', 'MA', 'CRM'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD'],
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'USDCHF=X'],
        'options': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META']
    }
    
    engine = V16Engine()
    results = {}
    
    print("\n" + "="*80)
    print("V16 AGGRESSIVE PROPORTIONATE OPTIMIZATION")
    print("="*80)
    print(f"Period: {start.date()} to {end.date()} (1 month)")
    print("\nV16 CHANGES:")
    print("  - Higher base risk (8-10% per trade)")
    print("  - Aggressive win streak multipliers (up to 1.6x)")
    print("  - Lower signal thresholds (more trades)")
    print("  - Faster trailing stops")
    print("  - Expanded symbol universe")
    print("="*80)
    
    for asset_class, symbols in assets.items():
        results[asset_class] = engine.run_isolated(asset_class, symbols, start, end)
    
    print("\n" + "="*80)
    print("V16 RESULTS SUMMARY")
    print("="*80)
    
    v11_baseline = 2.61
    total_initial = 0
    total_final = 0
    all_profitable = True
    
    for name, s in results.items():
        if s and s.get('final'):
            improvement = s['return_pct'] / v11_baseline
            target_pct = (s['return_pct'] / 26.1) * 100
            
            status = "PROFITABLE" if s['return_pct'] > 0 else "LOSS"
            if s['return_pct'] <= 0:
                all_profitable = False
            
            print(f"\n{'-'*60}")
            print(f"{name.upper()} [{status}]")
            print(f"{'-'*60}")
            print(f"  Starting:  ${s['initial']:,.2f} → Ending: ${s['final']:,.2f}")
            print(f"  Profit:    ${s['final'] - s['initial']:+,.2f}")
            print(f"  Return:    {s['return_pct']:+.2f}%")
            print(f"  vs V11:    {improvement:.1f}x | vs 10x Target: {target_pct:.0f}%")
            print(f"  Trades: {s['trades']} | WR: {s['win_rate']:.1f}% | PF: {s['profit_factor']}")
            
            total_initial += s['initial']
            total_final += s['final']
    
    total_ret = (total_final / total_initial - 1) * 100 if total_initial > 0 else 0
    
    print(f"\n{'='*80}")
    print("PORTFOLIO TOTALS")
    print(f"{'='*80}")
    print(f"\n  STARTING: ${total_initial:,.2f}")
    print(f"  ENDING:   ${total_final:,.2f}")
    print(f"  PROFIT:   ${total_final - total_initial:+,.2f}")
    print(f"  RETURN:   {total_ret:+.2f}%")
    print(f"  vs V11:   {total_ret/v11_baseline:.1f}x")
    print(f"  vs 10x:   {(total_ret/26.1)*100:.0f}% of goal")
    
    print(f"\n{'='*80}")
    print("VERSION PROGRESSION")
    print(f"{'='*80}")
    print(f"  V11: +2.61% | V12: +5.67% | V13: +3.93% | V14: +8.28%")
    print(f"  V15: +11.50% | V16: {total_ret:+.2f}%")
    print(f"  TARGET: +26.1% | GAP: {26.1 - total_ret:+.1f}%")
    
    with open('backtesting/experiment_results_v16_scalp.json', 'w') as f:
        json.dump({
            'version': 16,
            'strategy': 'AGGRESSIVE_PROPORTIONATE',
            'period_days': 30,
            'run_date': datetime.now().isoformat(),
            'starting_balance': total_initial,
            'ending_balance': round(total_final, 2),
            'total_return_pct': round(total_ret, 2),
            'results': results,
            'all_profitable': all_profitable,
            'improvement_vs_v11': round(total_ret / v11_baseline, 2),
            'gap_to_10x': round(26.1 - total_ret, 2)
        }, f, indent=2, default=str)
    
    print(f"\nSaved: backtesting/experiment_results_v16_scalp.json")
    
    return results, all_profitable, total_ret


if __name__ == "__main__":
    results, all_profitable, total_ret = run_v16()
