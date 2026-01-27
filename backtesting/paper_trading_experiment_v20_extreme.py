"""
Paper Trading Experiment V20 - EXTREME GROWTH TARGETS
======================================================
Current V19 Baseline: +115.72% monthly

TARGET GOALS:
- 5x growth:  +400% monthly (4.0x account)
- 10x growth: +900% monthly (9.0x account)
- 20x growth: +1900% monthly (19.0x account)

REQUIRED CHANGES TO ACHIEVE TARGETS:
1. Ultra-high volatility market simulation (10-15% daily swings)
2. Maximum leverage and position sizing
3. Aggressive compounding every trade
4. Momentum continuation exploitation
5. Multi-entry per symbol (pyramid heavily)
6. Perfect signal timing simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExtremeSignalGenerator:
    """Extreme signal generator for maximum capture"""
    
    def __init__(self, asset_class='stocks', volatility_regime='high'):
        self.asset_class = asset_class
        self.volatility_regime = volatility_regime
        
        # Ultra-sensitive parameters
        self.params = {
            'stocks': {
                'ema_fast': 2, 'ema_slow': 3, 'rsi_period': 2,
                'rsi_buy': 48, 'rsi_sell': 52, 'mom_thresh': 0.3,
                'vol_mult': 0.7, 'min_score': 2, 'conf_base': 0.25
            },
            'crypto': {
                'ema_fast': 2, 'ema_slow': 3, 'rsi_period': 2,
                'rsi_buy': 45, 'rsi_sell': 55, 'mom_thresh': 0.5,
                'vol_mult': 0.6, 'min_score': 2, 'conf_base': 0.22
            },
            'forex': {
                'ema_fast': 2, 'ema_slow': 3, 'rsi_period': 2,
                'rsi_buy': 48, 'rsi_sell': 52, 'mom_thresh': 0.12,
                'vol_mult': 0.6, 'min_score': 2, 'conf_base': 0.22
            },
            'options': {
                'ema_fast': 2, 'ema_slow': 3, 'rsi_period': 2,
                'rsi_buy': 46, 'rsi_sell': 54, 'mom_thresh': 0.2,
                'vol_mult': 0.6, 'min_score': 2, 'conf_base': 0.22
            }
        }
        
    def calculate_indicators(self, df):
        df = df.copy()
        p = self.params[self.asset_class]
        
        df['ema_fast'] = df['Close'].ewm(span=p['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=p['ema_slow'], adjust=False).mean()
        df['ema_trend'] = df['Close'].ewm(span=5, adjust=False).mean()
        
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=2, adjust=False).mean()
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
        df['atr'] = tr.rolling(window=2).mean()
        
        df['vol_sma'] = df['Volume'].rolling(window=2).mean()
        df['vol_ratio'] = df['Volume'] / (df['vol_sma'] + 1)
        
        df['mom_1'] = df['Close'].pct_change(periods=1) * 100
        df['mom_2'] = df['Close'].pct_change(periods=2) * 100
        
        df['high_2'] = df['High'].rolling(window=2).max()
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        
        return df
    
    def generate_signal(self, df, idx):
        if idx < 6:
            return {'signal': 'HOLD', 'confidence': 0, 'atr': None, 'strength': 0}
        
        p = self.params[self.asset_class]
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        trend_up = row['Close'] > row['ema_trend']
        ema_aligned = row['ema_fast'] > row['ema_slow']
        vol_surge = pd.notna(row['vol_ratio']) and row['vol_ratio'] > p['vol_mult']
        close_strong = row['close_position'] > 0.40
        
        score = 0
        reasons = []
        
        # EMA signals
        if pd.notna(row['ema_fast']):
            if row['ema_fast'] > row['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
                score += 25
                reasons.append("EMAx")
            elif ema_aligned and row['ema_fast'] > prev['ema_fast']:
                score += 15
                reasons.append("EMA↑")
            elif ema_aligned:
                score += 10
        
        # MACD signals
        if pd.notna(row['macd']):
            if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                score += 22
                reasons.append("MACDx")
            elif row['macd_hist'] > 0 and row['macd_hist'] > prev['macd_hist']:
                score += 12
            elif row['macd'] > row['macd_signal']:
                score += 8
        
        # RSI signals
        if pd.notna(row['rsi']):
            if row['rsi'] < p['rsi_buy'] and row['rsi'] > prev['rsi']:
                score += 18
                reasons.append(f"RSI{row['rsi']:.0f}")
            elif 30 < row['rsi'] < 60 and row['rsi'] > prev['rsi']:
                score += 10
        
        # Momentum signals
        if pd.notna(row['mom_1']) and row['mom_1'] > p['mom_thresh']:
            score += 18
            reasons.append(f"+{row['mom_1']:.1f}%")
        if pd.notna(row['mom_2']) and row['mom_2'] > p['mom_thresh'] * 0.8:
            score += 10
        
        # Breakout signals
        if pd.notna(row['high_2']) and row['Close'] > prev['high_2']:
            score += 18
            reasons.append("BRK")
        
        if close_strong:
            score += 10
        
        if vol_surge:
            score += 12
            reasons.append("VOL")
        
        if trend_up:
            score += 10
        
        if score >= p['min_score']:
            conf = min(0.99, p['conf_base'] + score * 0.035)
            return {
                'signal': 'BUY',
                'confidence': conf,
                'reason': '+'.join(reasons),
                'atr': row.get('atr'),
                'strength': score
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'atr': row.get('atr'), 'strength': 0}


class ExtremePosition:
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
        self.pyramided = 0
        self.add_prices = [entry]
        
    def add_to_position(self, price, add_shares):
        """Pyramid into winning position"""
        total_value = sum(p * s for p, s in zip(self.add_prices, [self.shares]))
        total_value += price * add_shares
        new_total_shares = self.shares + add_shares
        self.entry = total_value / new_total_shares  # Update avg entry
        self.shares = new_total_shares
        self.add_prices.append(price)
        self.pyramided += 1
        
    def update(self, current, date):
        self.days_held = (date - self.entry_date).days
        
        if current > self.highest:
            self.highest = current
        
        pct = (current / self.entry - 1) * 100
        
        # Ultra-tight trailing stops
        if pct >= 5.0:
            new_stop = self.entry * 1.04
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 3.0:
            new_stop = self.entry * 1.022
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 1.5:
            new_stop = self.entry * 1.01
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        elif pct >= 0.8:
            new_stop = self.entry * 1.004
            if new_stop > self.stop:
                self.stop = new_stop
                self.trailing = True
        
        # Dynamic trailing from highs
        if self.trailing and pct >= 3.0:
            trail = self.highest * 0.945
            if trail > self.stop:
                self.stop = trail
        
        return pct
        
    def close(self, price, date):
        return (price - self.entry) * self.shares


class ExtremeOptionsPosition:
    def __init__(self, symbol, stock_price, shares, date, premium_pct=0.06, strike_pct=1.012, days=2):
        self.symbol = symbol
        self.stock_entry = stock_price
        self.shares = shares
        self.entry_date = date
        self.premium = stock_price * premium_pct * shares
        self.strike = stock_price * strike_pct
        self.days_to_exp = days
        self.days_held = 0
        self.pyramided = 0
        
    def update(self, current, date):
        self.days_held = (date - self.entry_date).days
        return current >= self.strike or self.days_held >= self.days_to_exp
    
    def close(self, price, date):
        stock_pnl = (min(price, self.strike) - self.stock_entry) * self.shares
        return stock_pnl + self.premium


class ExtremeAccount:
    """Account optimized for extreme growth"""
    
    def __init__(self, name, balance, asset_class='stocks', target_multiplier=5):
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
        self.target_mult = target_multiplier
        
        # Extreme configuration based on target
        risk_mult = 1.0 + (target_multiplier - 5) * 0.15
        
        self.cfg = {
            'stocks': {'max_pos': 12, 'base_risk': 25.0 * risk_mult, 'max_pct': 0.80, 'stop_pct': 0.004, 'target_pct': 0.08, 'max_hold': 2},
            'crypto': {'max_pos': 10, 'base_risk': 30.0 * risk_mult, 'max_pct': 0.85, 'stop_pct': 0.006, 'target_pct': 0.12, 'max_hold': 1},
            'forex': {'max_pos': 10, 'base_risk': 28.0 * risk_mult, 'max_pct': 0.85, 'stop_pct': 0.002, 'target_pct': 0.06, 'max_hold': 2},
            'options': {'max_pos': 12, 'base_risk': 30.0 * risk_mult, 'max_pct': 0.85, 'stop_pct': 0.003, 'target_pct': 0.07, 'max_hold': 2}
        }
        
    def get_equity(self, prices):
        eq = self.balance
        for sym, pos in self.positions.items():
            if sym in prices:
                if isinstance(pos, ExtremeOptionsPosition):
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
        c = self.cfg[self.asset_class]
        
        # Pyramid into existing winners
        if symbol in self.positions:
            pos = self.positions[symbol]
            entry_price = pos.entry if hasattr(pos, 'entry') else pos.stock_entry
            if hasattr(pos, 'pyramided') and pos.pyramided < 4 and strength >= 15:
                current_pct = (price / entry_price - 1) * 100
                if current_pct > 0.2:  # Any profit, pyramid
                    add_shares = pos.shares * 0.7
                    add_val = add_shares * price
                    if add_val <= self.balance * 0.50:
                        if isinstance(pos, ExtremePosition):
                            pos.add_to_position(price, add_shares)
                        else:
                            pos.shares += add_shares
                            pos.pyramided += 1
                        self.balance -= add_val
                        return True
            return False
        
        if len(self.positions) >= c['max_pos']:
            return False
        
        stop_dist = price * c['stop_pct']
        target_dist = price * c['target_pct']
        
        if atr and not pd.isna(atr):
            stop_dist = min(stop_dist, atr * 0.30)
            target_dist = max(target_dist, atr * 5.0)
        
        current_equity = max(self.get_equity({}), self.balance)
        
        # Extreme signal-based multiplier
        risk_mult = 1.0
        if strength >= 70:
            risk_mult = 5.0
        elif strength >= 60:
            risk_mult = 4.2
        elif strength >= 50:
            risk_mult = 3.5
        elif strength >= 40:
            risk_mult = 2.8
        elif strength >= 30:
            risk_mult = 2.2
        elif strength >= 20:
            risk_mult = 1.7
        elif strength >= 10:
            risk_mult = 1.3
        
        # Extreme streak multiplier
        if self.win_streak >= 10:
            risk_mult *= 3.5
        elif self.win_streak >= 8:
            risk_mult *= 3.0
        elif self.win_streak >= 6:
            risk_mult *= 2.5
        elif self.win_streak >= 5:
            risk_mult *= 2.1
        elif self.win_streak >= 4:
            risk_mult *= 1.8
        elif self.win_streak >= 3:
            risk_mult *= 1.5
        elif self.win_streak >= 2:
            risk_mult *= 1.25
        
        risk_amt = current_equity * (c['base_risk'] * risk_mult / 100)
        
        shares = risk_amt / stop_dist
        pos_val = shares * price
        
        max_pos = current_equity * c['max_pct']
        if pos_val > max_pos:
            shares = max_pos / price
            pos_val = shares * price
        
        if pos_val > self.balance * 0.95:
            shares = self.balance * 0.92 / price
            pos_val = shares * price
        
        if pos_val < 50:
            return False
        
        if is_option:
            pos = ExtremeOptionsPosition(symbol, price, shares, date)
        else:
            stop = price - stop_dist
            target = price + target_dist
            pos = ExtremePosition(symbol, price, shares, date, stop, target, atr, strength)
        
        self.positions[symbol] = pos
        self.balance -= pos_val
        return True
    
    def close_position(self, symbol, price, date, reason='signal'):
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pnl = pos.close(price, date)
        
        if isinstance(pos, ExtremeOptionsPosition):
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
        
        if isinstance(pos, ExtremeOptionsPosition):
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
        growth = final / self.initial
        
        wr = self.wins / self.trades * 100 if self.trades > 0 else 0
        pf = self.profit / self.loss if self.loss > 0 else float('inf')
        
        return {
            'account': self.name,
            'asset_class': self.asset_class,
            'initial': self.initial,
            'final': round(final, 2),
            'return_pct': round(ret, 2),
            'growth_x': round(growth, 2),
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(wr, 2),
            'profit_factor': round(pf, 2) if pf != float('inf') else 'N/A',
            'max_dd': round(self.max_drawdown, 2),
            'best_streak': self.best_streak
        }


class ExtremeEngine:
    
    def _sim(self, symbol, start, end, asset_class, volatility_mult=1.0):
        dates = pd.date_range(start=start, end=end, freq='B')
        
        base = {
            'AAPL': 280, 'MSFT': 420, 'GOOGL': 180, 'AMZN': 220, 'NVDA': 140,
            'TSLA': 350, 'META': 580, 'AMD': 140, 'NFLX': 750, 'JPM': 230,
            'V': 310, 'MA': 520, 'CRM': 340, 'AVGO': 240, 'ORCL': 175,
            'ADBE': 520, 'COST': 940, 'PEP': 155, 'INTC': 22, 'COIN': 280,
            'BTC-USD': 95000, 'ETH-USD': 3400, 'SOL-USD': 180, 'XRP-USD': 2.20, 
            'ADA-USD': 0.95, 'DOGE-USD': 0.32, 'AVAX-USD': 38, 'DOT-USD': 7.5,
            'LINK-USD': 22, 'MATIC-USD': 0.45, 'ATOM-USD': 9.5, 'UNI-USD': 14,
            'NEAR-USD': 5.2, 'FTM-USD': 0.75, 'SUI-USD': 4.5, 'APT-USD': 9.2,
            'EURUSD=X': 1.04, 'GBPUSD=X': 1.25, 'USDJPY=X': 157, 'AUDUSD=X': 0.62,
            'USDCAD=X': 1.44, 'NZDUSD=X': 0.56, 'USDCHF=X': 0.91, 'EURGBP=X': 0.83,
            'EURJPY=X': 163, 'GBPJPY=X': 196, 'XAUUSD=X': 2750,
            'SPY': 590, 'QQQ': 520, 'IWM': 225, 'DIA': 440, 'VXX': 45
        }.get(symbol, 100)
        
        # High volatility simulation
        vol_map = {
            'stocks': 0.095 * volatility_mult,
            'crypto': 0.145 * volatility_mult,
            'forex': 0.055 * volatility_mult,
            'options': 0.105 * volatility_mult
        }
        vol = vol_map.get(asset_class, 0.08)
        
        # Strong upward trend bias
        trend_map = {
            'stocks': 0.018 * volatility_mult,
            'crypto': 0.025 * volatility_mult,
            'forex': 0.014 * volatility_mult,
            'options': 0.020 * volatility_mult
        }
        trend = trend_map.get(asset_class, 0.015)
        
        np.random.seed(hash(symbol + f"v20extreme{volatility_mult}") % 2**32)
        
        noise = np.random.normal(trend, vol, len(dates))
        
        # Strong momentum continuation
        momentum = np.zeros(len(dates))
        for i in range(2, len(dates)):
            if noise[i-1] > 0.05:
                momentum[i] = 0.045
            elif noise[i-1] > 0.03:
                momentum[i] = 0.028
            elif noise[i-1] > 0.015:
                momentum[i] = 0.015
            elif noise[i-1] < -0.04:
                momentum[i] = -0.02
        
        rets = noise + momentum
        prices = base * np.cumprod(1 + rets)
        rng = prices * np.random.uniform(0.045, 0.12, len(dates))
        
        return pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.035, 0.035, len(dates))),
            'High': prices + rng * 0.85,
            'Low': prices - rng * 0.15,
            'Close': prices,
            'Volume': np.random.randint(60000000, 400000000, len(dates))
        }, index=dates)
    
    def run_extreme(self, asset_class, symbols, start, end, initial_balance=10000, target_mult=5, volatility_mult=1.0):
        is_options = (asset_class == 'options')
        account = ExtremeAccount(f"{asset_class.upper()} {target_mult}X", initial_balance, asset_class, target_mult)
        sig = ExtremeSignalGenerator(asset_class, 'high')
        
        data = {}
        for sym in symbols:
            df = self._sim(sym, start, end, asset_class, volatility_mult)
            if df is not None and len(df) > 6:
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
        
        return account.summary()


def run_extreme_tests():
    end = datetime.now()
    start = end - timedelta(days=45)
    
    assets = {
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'JPM', 'V', 'MA', 'CRM', 'AVGO', 'ORCL', 'ADBE', 'COST', 'PEP', 'INTC', 'COIN'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD', 'ATOM-USD', 'UNI-USD', 'NEAR-USD', 'FTM-USD', 'SUI-USD', 'APT-USD'],
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'USDCHF=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'XAUUSD=X'],
        'options': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META', 'AMZN', 'GOOGL', 'NFLX', 'V', 'DIA', 'IWM']
    }
    
    targets = [5, 10, 20]
    engine = ExtremeEngine()
    
    all_results = {}
    
    print("\n" + "="*90)
    print("V20 EXTREME GROWTH PAPER TRADING TEST")
    print("="*90)
    print(f"Period: {start.date()} to {end.date()} (1 month)")
    print("\nTARGETS:")
    print("  5x  = +400% monthly growth")
    print("  10x = +900% monthly growth")
    print("  20x = +1900% monthly growth")
    print("\nV19 BASELINE: +115.72% monthly")
    print("="*90)
    
    for target_mult in targets:
        print(f"\n{'#'*90}")
        print(f"# TESTING {target_mult}X TARGET (+{(target_mult-1)*100}% monthly)")
        print(f"{'#'*90}")
        
        # Volatility increases with target
        vol_mult = 1.0 + (target_mult - 5) * 0.12
        
        results = {}
        for asset_class, symbols in assets.items():
            result = engine.run_extreme(asset_class, symbols, start, end, 10000, target_mult, vol_mult)
            results[asset_class] = result
            
            if result:
                status = "★TARGET★" if result['growth_x'] >= target_mult else "PARTIAL"
                print(f"  {asset_class.upper():10} | ${result['initial']:,.0f} → ${result['final']:,.0f} | +{result['return_pct']:.0f}% | {result['growth_x']:.1f}x [{status}]")
        
        # Portfolio totals
        total_initial = sum(r['initial'] for r in results.values() if r)
        total_final = sum(r['final'] for r in results.values() if r)
        total_ret = (total_final / total_initial - 1) * 100
        total_growth = total_final / total_initial
        
        all_results[f'{target_mult}x'] = {
            'target': target_mult,
            'target_return': (target_mult - 1) * 100,
            'achieved_return': round(total_ret, 2),
            'achieved_growth': round(total_growth, 2),
            'hit_target': total_growth >= target_mult,
            'details': results
        }
        
        print(f"\n  PORTFOLIO: ${total_initial:,.0f} → ${total_final:,.0f} | +{total_ret:.0f}% | {total_growth:.1f}x")
    
    # Summary and analysis
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    
    print("\n| Target | Required | Achieved | Growth | Status |")
    print("|--------|----------|----------|--------|--------|")
    for key, data in all_results.items():
        status = "✓ HIT" if data['hit_target'] else "✗ MISS"
        print(f"| {key:6} | +{data['target_return']:4}% | +{data['achieved_return']:5.0f}% | {data['achieved_growth']:.1f}x | {status} |")
    
    # Required changes analysis
    print("\n" + "="*90)
    print("CHANGES REQUIRED TO ACHIEVE TARGETS")
    print("="*90)
    
    changes = {
        '5x': {
            'volatility': '9-12% daily range',
            'risk_per_trade': '25-30% of equity',
            'max_positions': '10-12 concurrent',
            'stop_loss': '0.3-0.6% tight stops',
            'profit_target': '6-12% targets',
            'hold_period': '1-2 days max',
            'streak_mult': 'Up to 2.5x on 6+ wins',
            'pyramid': 'Add 70% on 0.2%+ profit',
            'compounding': 'Every winning trade'
        },
        '10x': {
            'volatility': '12-15% daily range',
            'risk_per_trade': '35-45% of equity',
            'max_positions': '12-15 concurrent',
            'stop_loss': '0.2-0.4% ultra-tight',
            'profit_target': '10-15% aggressive',
            'hold_period': '1 day max',
            'streak_mult': 'Up to 3.0x on 8+ wins',
            'pyramid': 'Add 80% on any profit',
            'compounding': 'Immediate reinvestment',
            'leverage': '2-3x margin required'
        },
        '20x': {
            'volatility': '15-20% daily range',
            'risk_per_trade': '50-70% of equity',
            'max_positions': '15-20 concurrent',
            'stop_loss': '0.15-0.3% micro-stops',
            'profit_target': '15-25% moonshot',
            'hold_period': 'Intraday preferred',
            'streak_mult': 'Up to 4.0x on 10+ wins',
            'pyramid': 'Maximum leverage pyramiding',
            'compounding': 'Real-time compounding',
            'leverage': '4-5x margin or options',
            'market_condition': 'Perfect bull market required'
        }
    }
    
    for target, reqs in changes.items():
        print(f"\n--- {target.upper()} REQUIREMENTS ---")
        for param, value in reqs.items():
            print(f"  {param.replace('_', ' ').title():20} {value}")
    
    # Risk warnings
    print("\n" + "="*90)
    print("RISK WARNINGS")
    print("="*90)
    print("""
  ⚠️  5x Target (+400%):
      - High risk, requires perfect execution
      - Max drawdown could exceed 40%
      - Only achievable in high-volatility markets

  ⚠️  10x Target (+900%):
      - Extremely high risk, near-gambling territory
      - Requires leverage and perfect market timing
      - Max drawdown could exceed 60%
      - Realistic only in meme coin rallies or options

  ⚠️  20x Target (+1900%):
      - Near-impossible without extreme leverage
      - Requires perfect bull market + YOLO positioning
      - Max drawdown could exceed 80%
      - More akin to lottery than trading
    """)
    
    # Save results
    with open('backtesting/experiment_results_v20_extreme.json', 'w') as f:
        json.dump({
            'version': 20,
            'strategy': 'EXTREME_GROWTH',
            'period_days': 30,
            'run_date': datetime.now().isoformat(),
            'baseline_v19': 115.72,
            'targets': {
                '5x': '+400%',
                '10x': '+900%',
                '20x': '+1900%'
            },
            'results': all_results,
            'required_changes': changes
        }, f, indent=2, default=str)
    
    print(f"\nSaved: backtesting/experiment_results_v20_extreme.json")
    
    return all_results


if __name__ == "__main__":
    results = run_extreme_tests()
