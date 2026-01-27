"""
Paper Trading Engine for IntelliTradeAI
Options-focused paper trading with risk management and improvement loops
"""

import os
import uuid
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import yfinance as yf
import numpy as np
from scipy.stats import norm

TARGET_SYMBOLS = ['GOOGL', 'TSM', 'NVDA', 'AMD', 'META', 'GEV', 'HOOD', 'V', 'MU', 'WDC', 'PLTR', 'LLY']

@dataclass
class OptionsPosition:
    position_id: str
    symbol: str
    option_type: str
    strike_price: float
    expiration_date: str
    contracts: int
    entry_price: float
    current_price: float = 0.0
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta: float = 0.0
    entry_vega: float = 0.0
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta: float = 0.0
    current_vega: float = 0.0
    implied_volatility: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    ai_signal: str = ""
    ai_confidence: float = 0.0
    status: str = "open"
    opened_at: str = ""
    closed_at: str = ""
    close_reason: str = ""

@dataclass
class TradingSession:
    session_id: str
    starting_balance: float = 100000.0
    current_balance: float = 100000.0
    target_balance: float = 200000.0
    peak_balance: float = 100000.0
    max_drawdown_limit: float = 30.0
    current_drawdown: float = 0.0
    status: str = "active"
    strategy_version: int = 1
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    started_at: str = ""
    ended_at: str = ""
    end_reason: str = ""
    positions: List[OptionsPosition] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)
    improvements: List[Dict] = field(default_factory=list)

@dataclass
class StrategyConfig:
    """V22 Optimized Strategy Configuration for 5x/10x Growth
    
    Note: This config is for OPTIONS trading engine.
    - stop_loss_percent/take_profit_percent are on UNDERLYING price movement
    - For V22 options: 0.25% stop on underlying = ~5% on premium
    - Imported from trading.v22_scalp_config for consistency
    """
    version: int = 22
    # V22 Position Sizing (Aggressive)
    position_size_percent: float = 32.0  # V22: 32% base risk for options
    max_positions: int = 8
    min_confidence: float = 45.0  # V22: Lower threshold for more trades
    # V22 Options: Using underlying price stops (not premium %)
    stop_loss_percent: float = 0.25  # V22: 0.25% stop on underlying
    take_profit_percent: float = 7.5  # V22: 7.5% target on underlying
    max_portfolio_exposure: float = 85.0  # V22: Allow 85% exposure
    max_days_to_expiry: int = 30  # V22: Shorter expiry for theta
    min_days_to_expiry: int = 7  # V22: Allow closer expiry
    delta_range_min: float = 0.45
    delta_range_max: float = 0.70
    calls_only: bool = True
    allowed_symbols: List[str] = field(default_factory=lambda: TARGET_SYMBOLS.copy())
    # V22 Scalping additions
    max_hold_days: int = 2
    pyramid_max: int = 5
    pyramid_add_percent: float = 75.0
    win_streak_multiplier: float = 3.8
    volatility_bonus_max: float = 1.35


class OptionsGreeksCalculator:
    """Black-Scholes Greeks Calculator"""
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0, S - K)
        d1, d2 = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0, K - S)
        d1, d2 = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1, _ = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            return 0.0
        d1, d2 = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        if option_type.lower() == 'call':
            term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
            return (term1 - term2) / 365
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1, _ = OptionsGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100


SYMBOL_METADATA = {
    'GOOGL': {'price': 195.0, 'iv': 0.28, 'vol': 0.02},
    'TSM': {'price': 205.0, 'iv': 0.32, 'vol': 0.025},
    'NVDA': {'price': 140.0, 'iv': 0.45, 'vol': 0.035},
    'AMD': {'price': 120.0, 'iv': 0.40, 'vol': 0.03},
    'META': {'price': 610.0, 'iv': 0.30, 'vol': 0.022},
    'GEV': {'price': 380.0, 'iv': 0.35, 'vol': 0.028},
    'HOOD': {'price': 48.0, 'iv': 0.55, 'vol': 0.04},
    'V': {'price': 320.0, 'iv': 0.22, 'vol': 0.015},
    'MU': {'price': 105.0, 'iv': 0.38, 'vol': 0.028},
    'WDC': {'price': 55.0, 'iv': 0.42, 'vol': 0.032},
    'PLTR': {'price': 75.0, 'iv': 0.50, 'vol': 0.038},
    'LLY': {'price': 780.0, 'iv': 0.25, 'vol': 0.018},
}


class SyntheticMarketSimulator:
    """Generate realistic synthetic options data with bullish trending"""
    
    def __init__(self, seed: int = None, bull_bias: float = 0.65):
        if seed:
            np.random.seed(seed)
        self.greeks_calc = OptionsGreeksCalculator()
        self.risk_free_rate = 0.05
        self.prices = {s: m['price'] for s, m in SYMBOL_METADATA.items()}
        self.cycle = 0
        self.bull_bias = bull_bias
        self.market_trend = 0.01
        self.trend_direction = 1
        self.trend_duration = 0
        self.bull_phase = True
    
    def advance_cycle(self):
        """Simulate bullish trending market"""
        self.cycle += 1
        self.trend_duration += 1
        
        if self.trend_duration > np.random.randint(15, 35):
            if self.bull_phase:
                self.bull_phase = np.random.random() < self.bull_bias
            else:
                self.bull_phase = np.random.random() < 0.7
            self.trend_direction = 1 if self.bull_phase else -1
            self.trend_duration = 0
        
        trend_strength = 0.008 * self.trend_direction
        self.market_trend = np.clip(
            self.market_trend * 0.85 + trend_strength + np.random.normal(0, 0.005),
            -0.03, 0.05
        )
        
        for symbol, meta in SYMBOL_METADATA.items():
            vol = meta['vol']
            base_drift = 0.002 if self.bull_phase else -0.001
            drift = self.market_trend + base_drift + np.random.normal(0, vol)
            self.prices[symbol] *= (1 + drift)
            self.prices[symbol] = max(1.0, self.prices[symbol])
    
    def get_stock_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)
    
    def generate_options_chain(self, symbol: str, target_dte: int = 30) -> Dict[str, Any]:
        stock_price = self.get_stock_price(symbol)
        meta = SYMBOL_METADATA.get(symbol, {'iv': 0.35})
        base_iv = meta['iv']
        
        exp_date = (datetime.now() + timedelta(days=target_dte)).strftime('%Y-%m-%d')
        T = target_dte / 365
        
        strikes = []
        step = max(1, round(stock_price * 0.025))
        for pct in range(-8, 9):
            strike = round(stock_price + pct * step)
            if strike > 0:
                strikes.append(float(strike))
        
        calls = []
        puts = []
        
        for strike in strikes:
            moneyness = stock_price / strike
            iv = base_iv * (1 + 0.1 * (1 - moneyness))
            iv = max(0.15, min(0.80, iv))
            
            call_price = self.greeks_calc.call_price(stock_price, strike, T, self.risk_free_rate, iv)
            put_price = self.greeks_calc.put_price(stock_price, strike, T, self.risk_free_rate, iv)
            
            spread = 0.02 + 0.01 * abs(1 - moneyness)
            
            calls.append({
                'strike': strike,
                'lastPrice': round(call_price, 2),
                'bid': round(call_price * (1 - spread/2), 2),
                'ask': round(call_price * (1 + spread/2), 2),
                'volume': int(np.random.exponential(500)),
                'openInterest': int(np.random.exponential(2000)),
                'impliedVolatility': iv,
                'delta': self.greeks_calc.delta(stock_price, strike, T, self.risk_free_rate, iv, 'call'),
                'gamma': self.greeks_calc.gamma(stock_price, strike, T, self.risk_free_rate, iv),
                'theta': self.greeks_calc.theta(stock_price, strike, T, self.risk_free_rate, iv, 'call'),
                'vega': self.greeks_calc.vega(stock_price, strike, T, self.risk_free_rate, iv),
                'inTheMoney': stock_price > strike
            })
            
            puts.append({
                'strike': strike,
                'lastPrice': round(put_price, 2),
                'bid': round(put_price * (1 - spread/2), 2),
                'ask': round(put_price * (1 + spread/2), 2),
                'volume': int(np.random.exponential(400)),
                'openInterest': int(np.random.exponential(1500)),
                'impliedVolatility': iv,
                'delta': self.greeks_calc.delta(stock_price, strike, T, self.risk_free_rate, iv, 'put'),
                'gamma': self.greeks_calc.gamma(stock_price, strike, T, self.risk_free_rate, iv),
                'theta': self.greeks_calc.theta(stock_price, strike, T, self.risk_free_rate, iv, 'put'),
                'vega': self.greeks_calc.vega(stock_price, strike, T, self.risk_free_rate, iv),
                'inTheMoney': stock_price < strike
            })
        
        return {
            'calls': calls,
            'puts': puts,
            'expiration': exp_date,
            'stock_price': stock_price
        }


class OptionsDataService:
    """Fetch options data and calculate Greeks with persistent caching"""
    
    CACHE_DIR = "data/options_cache"
    CACHE_TTL_MINUTES = 60
    
    def __init__(self, offline_mode: bool = False, synthetic_mode: bool = False):
        self.greeks_calc = OptionsGreeksCalculator()
        self.risk_free_rate = 0.05
        self.cache = {}
        self.cache_expiry = {}
        self.offline_mode = offline_mode
        self.synthetic_mode = synthetic_mode
        self.synthetic_sim = SyntheticMarketSimulator() if synthetic_mode else None
        self.last_api_call = 0
        self.min_call_interval = 1.0
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_persistent_cache()
    
    def _get_cache_path(self, symbol: str) -> str:
        return os.path.join(self.CACHE_DIR, f"{symbol}.json")
    
    def _load_persistent_cache(self):
        """Load cached data from disk"""
        for filename in os.listdir(self.CACHE_DIR) if os.path.exists(self.CACHE_DIR) else []:
            if filename.endswith('.json'):
                symbol = filename[:-5]
                try:
                    with open(os.path.join(self.CACHE_DIR, filename), 'r') as f:
                        data = json.load(f)
                        cache_key = f"{symbol}_30"
                        self.cache[cache_key] = data.get('data', {})
                        cached_time = datetime.fromisoformat(data.get('timestamp', '2020-01-01'))
                        self.cache_expiry[cache_key] = cached_time + timedelta(minutes=self.CACHE_TTL_MINUTES)
                except Exception:
                    pass
    
    def _save_to_disk(self, symbol: str, data: Dict):
        """Persist cache to disk"""
        try:
            cache_path = self._get_cache_path(symbol)
            with open(cache_path, 'w') as f:
                json.dump({'data': data, 'timestamp': datetime.now().isoformat()}, f)
        except Exception:
            pass
    
    def _rate_limit(self):
        """Simple rate limiting - 1 call per second"""
        import time
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_api_call = time.time()
    
    def get_stock_price(self, symbol: str) -> float:
        if self.synthetic_mode and self.synthetic_sim:
            return self.synthetic_sim.get_stock_price(symbol)
        
        cache_key = f"{symbol}_30"
        if cache_key in self.cache and self.cache[cache_key].get('stock_price'):
            return self.cache[cache_key]['stock_price']
        
        if self.offline_mode:
            return SYMBOL_METADATA.get(symbol, {}).get('price', 100.0)
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            if '429' in str(e) or 'Rate' in str(e):
                if cache_key in self.cache:
                    return self.cache[cache_key].get('stock_price', 100.0)
        return 100.0
    
    def get_options_chain(self, symbol: str, target_dte: int = 30) -> Dict[str, Any]:
        if self.synthetic_mode and self.synthetic_sim:
            return self.synthetic_sim.generate_options_chain(symbol, target_dte)
        
        cache_key = f"{symbol}_{target_dte}"
        now = datetime.now()
        
        if cache_key in self.cache and cache_key in self.cache_expiry:
            if now < self.cache_expiry[cache_key]:
                return self.cache[cache_key]
        
        if self.offline_mode and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                if cache_key in self.cache:
                    return self.cache[cache_key]
                return {'calls': [], 'puts': [], 'expiration': None, 'stock_price': 0}
            
            target_date = now + timedelta(days=target_dte)
            best_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
            
            self._rate_limit()
            chain = ticker.option_chain(best_exp)
            stock_price = self.get_stock_price(symbol)
            
            calls = self._process_options(chain.calls, stock_price, best_exp, 'call')
            puts = self._process_options(chain.puts, stock_price, best_exp, 'put')
            
            result = {
                'calls': calls,
                'puts': puts,
                'expiration': best_exp,
                'stock_price': stock_price
            }
            
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = now + timedelta(minutes=self.CACHE_TTL_MINUTES)
            self._save_to_disk(symbol, result)
            
            return result
            
        except Exception as e:
            if cache_key in self.cache:
                return self.cache[cache_key]
            return {'calls': [], 'puts': [], 'expiration': None, 'stock_price': 0}
    
    def _process_options(self, options_df, stock_price: float, expiration: str, option_type: str) -> List[Dict]:
        if options_df is None or options_df.empty:
            return []
        
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        T = max((exp_date - datetime.now()).days / 365, 0.001)
        
        processed = []
        for _, row in options_df.iterrows():
            strike = float(row['strike'])
            iv = float(row.get('impliedVolatility', 0))
            if iv < 0.10:
                iv = 0.35
            elif iv > 2.0:
                iv = 0.60
            last_price = float(row.get('lastPrice', 0))
            bid = float(row.get('bid', 0))
            ask = float(row.get('ask', 0))
            volume = int(row.get('volume', 0)) if not np.isnan(row.get('volume', 0)) else 0
            open_interest = int(row.get('openInterest', 0)) if not np.isnan(row.get('openInterest', 0)) else 0
            
            delta = self.greeks_calc.delta(stock_price, strike, T, self.risk_free_rate, iv, option_type)
            gamma = self.greeks_calc.gamma(stock_price, strike, T, self.risk_free_rate, iv)
            theta = self.greeks_calc.theta(stock_price, strike, T, self.risk_free_rate, iv, option_type)
            vega = self.greeks_calc.vega(stock_price, strike, T, self.risk_free_rate, iv)
            
            processed.append({
                'strike': strike,
                'lastPrice': last_price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'openInterest': open_interest,
                'impliedVolatility': iv,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'inTheMoney': (stock_price > strike) if option_type == 'call' else (stock_price < strike)
            })
        
        return processed
    
    def select_optimal_option(self, symbol: str, signal: str, stock_price: float, 
                              strategy: StrategyConfig) -> Optional[Dict]:
        chain = self.get_options_chain(symbol, target_dte=30)
        
        if not chain['calls'] and not chain['puts']:
            return None
        
        options = chain['calls'] if signal == 'BUY' else chain['puts']
        option_type = 'call' if signal == 'BUY' else 'put'
        
        suitable = []
        for opt in options:
            delta = abs(opt['delta'])
            if strategy.delta_range_min <= delta <= strategy.delta_range_max:
                price = opt['lastPrice']
                if opt['bid'] > 0 and opt['ask'] > 0:
                    price = (opt['bid'] + opt['ask']) / 2
                    spread = (opt['ask'] - opt['bid']) / opt['ask']
                    if spread < 0.20:
                        opt['_price'] = price
                        suitable.append(opt)
                elif price > 0:
                    opt['_price'] = price
                    suitable.append(opt)
        
        if not suitable:
            return None
        
        suitable.sort(key=lambda x: abs(abs(x['delta']) - 0.50))
        best = suitable[0]
        
        return {
            'symbol': symbol,
            'option_type': option_type,
            'strike': best['strike'],
            'expiration': chain['expiration'],
            'price': best['_price'],
            'delta': best['delta'],
            'gamma': best['gamma'],
            'theta': best['theta'],
            'vega': best['vega'],
            'iv': best['impliedVolatility'],
            'stock_price': chain['stock_price']
        }


class PaperTradingEngine:
    """Main Paper Trading Engine with risk management"""
    
    def __init__(self, starting_balance: float = 100000.0, 
                 target_balance: float = 200000.0,
                 max_drawdown: float = 30.0,
                 offline_mode: bool = False,
                 synthetic_mode: bool = False):
        self.synthetic_mode = synthetic_mode
        self.options_service = OptionsDataService(offline_mode=offline_mode, synthetic_mode=synthetic_mode)
        self.strategy = StrategyConfig()
        self.session: Optional[TradingSession] = None
        self.starting_balance = starting_balance
        self.target_balance = target_balance
        self.max_drawdown = max_drawdown
        self.is_running = False
        self.cache_warmed = False
    
    def warmup_cache(self) -> int:
        """Pre-fetch data for all symbols to avoid rate limits during training"""
        import time
        fetched = 0
        print("Warming up cache for all symbols...")
        for symbol in self.strategy.allowed_symbols:
            try:
                chain = self.options_service.get_options_chain(symbol)
                if chain.get('calls'):
                    fetched += 1
                    print(f"  Cached {symbol}: {len(chain['calls'])} calls, ${chain['stock_price']:.2f}")
                time.sleep(2)
            except Exception as e:
                print(f"  Failed {symbol}: {e}")
        self.cache_warmed = True
        self.options_service.offline_mode = True
        print(f"Cache warmup complete: {fetched}/{len(self.strategy.allowed_symbols)} symbols")
        return fetched
        
    def start_session(self) -> TradingSession:
        session_id = str(uuid.uuid4())
        self.session = TradingSession(
            session_id=session_id,
            starting_balance=self.starting_balance,
            current_balance=self.starting_balance,
            target_balance=self.target_balance,
            peak_balance=self.starting_balance,
            max_drawdown_limit=self.max_drawdown,
            strategy_version=self.strategy.version,
            started_at=datetime.now().isoformat()
        )
        self.is_running = True
        print(f"Started paper trading session {session_id}")
        print(f"Starting balance: ${self.starting_balance:,.2f}")
        print(f"Target: ${self.target_balance:,.2f}")
        print(f"Max drawdown: {self.max_drawdown}%")
        return self.session
    
    def get_ai_signals(self) -> List[Dict]:
        signals = []
        for symbol in self.strategy.allowed_symbols:
            try:
                if self.synthetic_mode and self.options_service.synthetic_sim:
                    current_price = self.options_service.synthetic_sim.get_stock_price(symbol)
                    base_price = SYMBOL_METADATA.get(symbol, {}).get('price', 100.0)
                    pct_change = ((current_price - base_price) / base_price) * 100
                    rsi = 50 + np.random.normal(0, 15)
                    rsi = max(10, min(90, rsi))
                    sma_20 = base_price * (1 + np.random.normal(0, 0.02))
                else:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='30d')
                    
                    if hist.empty or len(hist) < 14:
                        continue
                    
                    close_prices = hist['Close'].values
                    delta = close_prices[-1] - close_prices[-5]
                    pct_change = (delta / close_prices[-5]) * 100 if close_prices[-5] > 0 else 0
                    
                    rsi = self._calculate_rsi(close_prices, 14)
                    
                    sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.mean(close_prices)
                    current_price = close_prices[-1]
                
                signal = None
                confidence = 0
                
                trend_up = current_price > sma_20
                momentum_up = pct_change > 0
                
                if rsi < 25:
                    signal = 'BUY'
                    confidence = min(90, 65 + (25 - rsi) * 2)
                elif rsi > 75 and not getattr(self.strategy, 'calls_only', False):
                    signal = 'SELL'
                    confidence = min(90, 65 + (rsi - 75) * 2)
                elif rsi < 40 and trend_up and momentum_up:
                    signal = 'BUY'
                    confidence = min(85, 55 + (40 - rsi) * 1.2)
                elif rsi > 60 and not trend_up and not momentum_up and not getattr(self.strategy, 'calls_only', False):
                    signal = 'SELL'
                    confidence = min(85, 55 + (rsi - 60) * 1.2)
                elif pct_change > 4 and rsi < 55:
                    signal = 'BUY'
                    confidence = min(80, 50 + pct_change * 3)
                elif pct_change < -4 and rsi > 45 and not getattr(self.strategy, 'calls_only', False):
                    signal = 'SELL'
                    confidence = min(80, 50 + abs(pct_change) * 3)
                
                if signal and confidence >= self.strategy.min_confidence:
                    signals.append({
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'current_price': current_price,
                        'rsi': rsi,
                        'sma_20': sma_20,
                        'pct_change_5d': pct_change
                    })
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        if not self.session or self.session.status != 'active':
            return None
        
        if len([p for p in self.session.positions if p.status == 'open']) >= self.strategy.max_positions:
            return {'success': False, 'reason': 'Max positions reached'}
        
        open_positions = [p for p in self.session.positions if p.status == 'open']
        current_exposure = sum(p.entry_price * p.contracts * 100 for p in open_positions)
        positions_value = sum(p.current_price * p.contracts * 100 for p in open_positions)
        current_equity = self.session.current_balance + positions_value
        exposure_pct = (current_exposure / current_equity) * 100 if current_equity > 0 else 100
        max_exposure = getattr(self.strategy, 'max_portfolio_exposure', 60)
        if exposure_pct >= max_exposure:
            return {'success': False, 'reason': f'Portfolio exposure at {exposure_pct:.0f}%'}
        
        option = self.options_service.select_optimal_option(
            signal['symbol'], 
            signal['signal'],
            signal['current_price'],
            self.strategy
        )
        
        if not option:
            return {'success': False, 'reason': 'No suitable option found'}
        
        position_value = self.session.current_balance * (self.strategy.position_size_percent / 100)
        contracts = max(1, int(position_value / (option['price'] * 100)))
        
        total_cost = contracts * option['price'] * 100
        fees = total_cost * 0.001
        
        if total_cost + fees > self.session.current_balance:
            contracts = max(1, int((self.session.current_balance * 0.9) / (option['price'] * 100)))
            total_cost = contracts * option['price'] * 100
            fees = total_cost * 0.001
            
            if total_cost + fees > self.session.current_balance:
                return {'success': False, 'reason': 'Insufficient funds'}
        
        position_id = str(uuid.uuid4())
        position = OptionsPosition(
            position_id=position_id,
            symbol=signal['symbol'],
            option_type=option['option_type'],
            strike_price=option['strike'],
            expiration_date=option['expiration'],
            contracts=contracts,
            entry_price=option['price'],
            current_price=option['price'],
            entry_delta=option['delta'],
            entry_gamma=option['gamma'],
            entry_theta=option['theta'],
            entry_vega=option['vega'],
            current_delta=option['delta'],
            current_gamma=option['gamma'],
            current_theta=option['theta'],
            current_vega=option['vega'],
            implied_volatility=option['iv'],
            ai_signal=signal['signal'],
            ai_confidence=signal['confidence'],
            opened_at=datetime.now().isoformat()
        )
        
        self.session.positions.append(position)
        self.session.current_balance -= (total_cost + fees)
        self.session.total_trades += 1
        
        trade_record = {
            'trade_id': str(uuid.uuid4()),
            'position_id': position_id,
            'symbol': signal['symbol'],
            'action': 'BUY',
            'option_type': option['option_type'],
            'strike': option['strike'],
            'expiration': option['expiration'],
            'contracts': contracts,
            'price': option['price'],
            'total_value': total_cost,
            'fees': fees,
            'ai_signal': signal['signal'],
            'ai_confidence': signal['confidence'],
            'executed_at': datetime.now().isoformat()
        }
        self.session.trade_history.append(trade_record)
        
        print(f"Opened position: {contracts}x {signal['symbol']} {option['option_type'].upper()} ${option['strike']} exp {option['expiration']}")
        print(f"  Entry: ${option['price']:.2f}, Total: ${total_cost:.2f}, Delta: {option['delta']:.3f}")
        
        return {'success': True, 'position': asdict(position), 'trade': trade_record}
    
    def update_positions(self) -> Dict[str, Any]:
        if not self.session:
            return {}
        
        total_unrealized = 0.0
        positions_value = 0.0
        
        for position in self.session.positions:
            if position.status != 'open':
                continue
            
            chain = self.options_service.get_options_chain(position.symbol)
            options = chain['calls'] if position.option_type == 'call' else chain['puts']
            
            current_opt = None
            for opt in options:
                if abs(opt['strike'] - position.strike_price) < 0.01:
                    current_opt = opt
                    break
            
            if current_opt:
                if current_opt['bid'] > 0 and current_opt['ask'] > 0:
                    position.current_price = (current_opt['bid'] + current_opt['ask']) / 2
                elif current_opt['lastPrice'] > 0:
                    position.current_price = current_opt['lastPrice']
                position.current_delta = current_opt['delta']
                position.current_gamma = current_opt['gamma']
                position.current_theta = current_opt['theta']
                position.current_vega = current_opt['vega']
            
            if position.current_price <= 0:
                position.current_price = position.entry_price
            
            if self.synthetic_mode and self.options_service.synthetic_sim:
                stock_price = self.options_service.synthetic_sim.get_stock_price(position.symbol)
                base_price = SYMBOL_METADATA.get(position.symbol, {}).get('price', stock_price)
                stock_move = (stock_price - base_price) / base_price
                delta_impact = stock_move * abs(position.current_delta) * 0.5
                theta_decay = position.current_theta / position.entry_price if position.entry_price > 0 else 0
                price_change = delta_impact + theta_decay
                position.current_price = position.entry_price * (1 + price_change)
            position.current_price = max(0.01, position.current_price)
            
            current_value = position.current_price * position.contracts * 100
            entry_value = position.entry_price * position.contracts * 100
            position.unrealized_pnl = current_value - entry_value
            
            total_unrealized += position.unrealized_pnl
            positions_value += current_value
            
            pnl_percent = (position.unrealized_pnl / entry_value) * 100 if entry_value > 0 else 0
            
            if pnl_percent <= -self.strategy.stop_loss_percent:
                self._close_position(position, 'stop_loss')
            elif pnl_percent >= self.strategy.take_profit_percent:
                self._close_position(position, 'take_profit')
            
            exp_date = datetime.strptime(position.expiration_date, '%Y-%m-%d')
            days_to_exp = (exp_date - datetime.now()).days
            if days_to_exp <= 2:
                self._close_position(position, 'expiration_approaching')
        
        self.session.total_unrealized_pnl = total_unrealized
        
        portfolio_value = self.session.current_balance + positions_value
        
        if portfolio_value > self.session.peak_balance:
            self.session.peak_balance = portfolio_value
        
        drawdown = ((self.session.peak_balance - portfolio_value) / self.session.peak_balance) * 100
        self.session.current_drawdown = drawdown
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.session.current_balance,
            'positions_value': positions_value,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': self.session.total_realized_pnl,
            'drawdown': drawdown,
            'peak_balance': self.session.peak_balance
        }
    
    def _close_position(self, position: OptionsPosition, reason: str):
        if position.status != 'open':
            return
        
        current_value = position.current_price * position.contracts * 100
        entry_value = position.entry_price * position.contracts * 100
        realized_pnl = current_value - entry_value
        fees = current_value * 0.001
        realized_pnl -= fees
        
        position.realized_pnl = realized_pnl
        position.status = 'closed'
        position.closed_at = datetime.now().isoformat()
        position.close_reason = reason
        
        self.session.current_balance += (current_value - fees)
        self.session.total_realized_pnl += realized_pnl
        
        if realized_pnl >= 0:
            self.session.winning_trades += 1
        else:
            self.session.losing_trades += 1
        
        trade_record = {
            'trade_id': str(uuid.uuid4()),
            'position_id': position.position_id,
            'symbol': position.symbol,
            'action': 'SELL',
            'option_type': position.option_type,
            'strike': position.strike_price,
            'expiration': position.expiration_date,
            'contracts': position.contracts,
            'price': position.current_price,
            'total_value': current_value,
            'fees': fees,
            'realized_pnl': realized_pnl,
            'close_reason': reason,
            'executed_at': datetime.now().isoformat()
        }
        self.session.trade_history.append(trade_record)
        
        print(f"Closed position ({reason}): {position.symbol} {position.option_type.upper()}")
        print(f"  P&L: ${realized_pnl:,.2f}")
    
    def check_session_status(self) -> Dict[str, Any]:
        if not self.session:
            return {'status': 'no_session'}
        
        status = self.update_positions()
        portfolio_value = status.get('portfolio_value', self.session.current_balance)
        
        if portfolio_value >= self.session.target_balance:
            self._end_session('target_reached')
            return {
                'status': 'target_reached',
                'message': f'Target of ${self.session.target_balance:,.2f} reached!',
                'final_value': portfolio_value,
                'total_return': ((portfolio_value - self.session.starting_balance) / self.session.starting_balance) * 100
            }
        
        if self.session.current_drawdown >= self.session.max_drawdown_limit:
            analysis = self._analyze_failure()
            self._end_session('drawdown_exceeded')
            return {
                'status': 'drawdown_exceeded',
                'message': f'Max drawdown of {self.session.max_drawdown_limit}% exceeded',
                'final_value': portfolio_value,
                'drawdown': self.session.current_drawdown,
                'analysis': analysis,
                'improvements': self._generate_improvements(analysis)
            }
        
        return {
            'status': 'active',
            'portfolio_value': portfolio_value,
            'drawdown': self.session.current_drawdown,
            'return_pct': ((portfolio_value - self.session.starting_balance) / self.session.starting_balance) * 100,
            'target_progress': ((portfolio_value - self.session.starting_balance) / 
                               (self.session.target_balance - self.session.starting_balance)) * 100
        }
    
    def _analyze_failure(self) -> Dict[str, Any]:
        closed_positions = [p for p in self.session.positions if p.status == 'closed']
        
        win_rate = (self.session.winning_trades / self.session.total_trades * 100) if self.session.total_trades > 0 else 0
        
        winning_pnl = sum(p.realized_pnl for p in closed_positions if p.realized_pnl > 0)
        losing_pnl = abs(sum(p.realized_pnl for p in closed_positions if p.realized_pnl < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        avg_win = winning_pnl / self.session.winning_trades if self.session.winning_trades > 0 else 0
        avg_loss = losing_pnl / self.session.losing_trades if self.session.losing_trades > 0 else 0
        
        symbol_performance = {}
        for p in closed_positions:
            if p.symbol not in symbol_performance:
                symbol_performance[p.symbol] = {'pnl': 0, 'trades': 0, 'wins': 0}
            symbol_performance[p.symbol]['pnl'] += p.realized_pnl
            symbol_performance[p.symbol]['trades'] += 1
            if p.realized_pnl > 0:
                symbol_performance[p.symbol]['wins'] += 1
        
        losing_symbols = [s for s, data in symbol_performance.items() if data['pnl'] < 0]
        
        return {
            'total_trades': self.session.total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.session.current_drawdown,
            'symbol_performance': symbol_performance,
            'losing_symbols': losing_symbols,
            'strategy_version': self.session.strategy_version
        }
    
    def _generate_improvements(self, analysis: Dict) -> Dict[str, Any]:
        improvements = {
            'strategy_changes': [],
            'new_parameters': {},
            'symbols_to_exclude': [],
            'version': self.strategy.version + 1
        }
        
        if analysis['total_pnl'] < 0:
            improvements['strategy_changes'].append('Reduce position size after loss')
            improvements['new_parameters']['position_size_percent'] = max(5, self.strategy.position_size_percent * 0.85)
        
        if analysis['win_rate'] < 45:
            improvements['strategy_changes'].append('Increase confidence threshold')
            improvements['new_parameters']['min_confidence'] = min(65, self.strategy.min_confidence + 5)
        
        if analysis['profit_factor'] < 1.0:
            improvements['strategy_changes'].append('Widen take profit')
            improvements['new_parameters']['take_profit_percent'] = min(50, self.strategy.take_profit_percent + 5)
        
        if analysis['avg_loss'] > analysis['avg_win'] * 1.3:
            improvements['strategy_changes'].append('Tighten stop loss')
            improvements['new_parameters']['stop_loss_percent'] = max(8, self.strategy.stop_loss_percent - 2)
        
        for symbol in analysis['losing_symbols']:
            data = analysis['symbol_performance'][symbol]
            if data['trades'] >= 2 and data['wins'] / data['trades'] < 0.35:
                improvements['symbols_to_exclude'].append(symbol)
                improvements['strategy_changes'].append(f'Exclude {symbol}')
        
        return improvements
    
    def apply_improvements(self, improvements: Dict) -> StrategyConfig:
        self.strategy.version = improvements.get('version', self.strategy.version + 1)
        
        for param, value in improvements.get('new_parameters', {}).items():
            if hasattr(self.strategy, param):
                setattr(self.strategy, param, value)
        
        for symbol in improvements.get('symbols_to_exclude', []):
            if symbol in self.strategy.allowed_symbols:
                self.strategy.allowed_symbols.remove(symbol)
        
        improvement_record = {
            'improvement_id': str(uuid.uuid4()),
            'session_id': self.session.session_id if self.session else None,
            'trigger_reason': 'drawdown_exceeded',
            'improvements_made': improvements,
            'new_strategy_version': self.strategy.version,
            'created_at': datetime.now().isoformat()
        }
        
        if self.session:
            self.session.improvements.append(improvement_record)
        
        print(f"Applied improvements - Strategy v{self.strategy.version}")
        for change in improvements.get('strategy_changes', []):
            print(f"  - {change}")
        
        return self.strategy
    
    def _end_session(self, reason: str):
        if not self.session:
            return
        
        for position in self.session.positions:
            if position.status == 'open':
                self._close_position(position, 'session_ended')
        
        self.session.status = 'ended'
        self.session.ended_at = datetime.now().isoformat()
        self.session.end_reason = reason
        self.is_running = False
        
        print(f"\nSession ended: {reason}")
        print(f"Final balance: ${self.session.current_balance:,.2f}")
        print(f"Total P&L: ${self.session.total_realized_pnl:,.2f}")
        print(f"Win rate: {self.session.winning_trades}/{self.session.total_trades}")
    
    def restart_with_improvements(self) -> TradingSession:
        if self.session and self.session.status == 'ended':
            if self.session.end_reason == 'drawdown_exceeded':
                analysis = self._analyze_failure()
                improvements = self._generate_improvements(analysis)
                self.apply_improvements(improvements)
        
        return self.start_session()
    
    def get_session_summary(self) -> Dict[str, Any]:
        if not self.session:
            return {}
        
        status = self.update_positions()
        open_positions = [asdict(p) for p in self.session.positions if p.status == 'open']
        closed_positions = [asdict(p) for p in self.session.positions if p.status == 'closed']
        
        return {
            'session_id': self.session.session_id,
            'status': self.session.status,
            'strategy_version': self.session.strategy_version,
            'starting_balance': self.session.starting_balance,
            'current_balance': self.session.current_balance,
            'target_balance': self.session.target_balance,
            'portfolio_value': status.get('portfolio_value', self.session.current_balance),
            'peak_balance': self.session.peak_balance,
            'current_drawdown': self.session.current_drawdown,
            'max_drawdown_limit': self.session.max_drawdown_limit,
            'total_realized_pnl': self.session.total_realized_pnl,
            'total_unrealized_pnl': self.session.total_unrealized_pnl,
            'total_trades': self.session.total_trades,
            'winning_trades': self.session.winning_trades,
            'losing_trades': self.session.losing_trades,
            'win_rate': (self.session.winning_trades / self.session.total_trades * 100) if self.session.total_trades > 0 else 0,
            'return_pct': ((status.get('portfolio_value', self.session.current_balance) - self.session.starting_balance) / self.session.starting_balance) * 100,
            'open_positions': open_positions,
            'closed_positions': closed_positions,
            'trade_history': self.session.trade_history,
            'improvements': self.session.improvements,
            'started_at': self.session.started_at,
            'ended_at': self.session.ended_at,
            'end_reason': self.session.end_reason
        }
    
    def run_trading_cycle(self) -> Dict[str, Any]:
        if not self.session or self.session.status != 'active':
            return {'error': 'No active session'}
        
        if self.synthetic_mode and self.options_service.synthetic_sim:
            self.options_service.synthetic_sim.advance_cycle()
        
        status = self.check_session_status()
        if status['status'] != 'active':
            return status
        
        signals = self.get_ai_signals()
        
        executed = []
        for signal in signals[:3]:
            if len([p for p in self.session.positions if p.status == 'open']) >= self.strategy.max_positions:
                break
            
            existing = [p for p in self.session.positions 
                       if p.symbol == signal['symbol'] and p.status == 'open']
            if existing:
                continue
            
            result = self.execute_signal(signal)
            if result and result.get('success'):
                executed.append(result)
        
        status = self.check_session_status()
        
        snapshot = {
            'snapshot_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': status.get('portfolio_value', self.session.current_balance),
            'cash_balance': self.session.current_balance,
            'unrealized_pnl': self.session.total_unrealized_pnl,
            'realized_pnl': self.session.total_realized_pnl,
            'drawdown': self.session.current_drawdown,
            'open_positions': len([p for p in self.session.positions if p.status == 'open'])
        }
        self.session.snapshots.append(snapshot)
        
        return {
            'status': status,
            'signals_found': len(signals),
            'trades_executed': len(executed),
            'executed_trades': executed,
            'snapshot': snapshot
        }


def run_paper_trading_bot():
    """Main function to run the paper trading bot"""
    engine = PaperTradingEngine(
        starting_balance=100000.0,
        target_balance=200000.0,
        max_drawdown=30.0
    )
    
    session_count = 0
    max_sessions = 10
    
    while session_count < max_sessions:
        session_count += 1
        print(f"\n{'='*60}")
        print(f"STARTING SESSION #{session_count}")
        print(f"{'='*60}\n")
        
        engine.start_session()
        
        cycle_count = 0
        max_cycles = 100
        
        while engine.is_running and cycle_count < max_cycles:
            cycle_count += 1
            print(f"\n--- Trading Cycle {cycle_count} ---")
            
            result = engine.run_trading_cycle()
            
            if result.get('status', {}).get('status') == 'target_reached':
                print("\n" + "="*60)
                print("SUCCESS! Target balance reached!")
                print("="*60)
                summary = engine.get_session_summary()
                print(f"Final Portfolio Value: ${summary['portfolio_value']:,.2f}")
                print(f"Total Return: {summary['return_pct']:.2f}%")
                print(f"Win Rate: {summary['win_rate']:.1f}%")
                return summary
            
            if result.get('status', {}).get('status') == 'drawdown_exceeded':
                print("\n" + "="*60)
                print("DRAWDOWN EXCEEDED - Analyzing and improving...")
                print("="*60)
                
                improvements = result.get('improvements', {})
                engine.apply_improvements(improvements)
                break
            
            print(f"Portfolio: ${result.get('status', {}).get('portfolio_value', 0):,.2f}")
            print(f"Drawdown: {result.get('status', {}).get('drawdown', 0):.2f}%")
            print(f"Progress: {result.get('status', {}).get('target_progress', 0):.1f}%")
        
        if engine.is_running:
            print("Max cycles reached, restarting session...")
            engine._end_session('max_cycles')
    
    print(f"\nMax sessions ({max_sessions}) reached without hitting target.")
    return engine.get_session_summary()


if __name__ == "__main__":
    run_paper_trading_bot()
