"""
Crypto Paper Trading Engine for IntelliTradeAI
Aggressive crypto trading simulator with synthetic market data
"""

import os
import uuid
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

TOP_CRYPTOS = [
    'BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'AVAX', 'SHIB', 'DOT', 'LINK',
    'MATIC', 'UNI', 'ATOM', 'LTC', 'XLM', 'NEAR', 'APT', 'ARB', 'OP', 'INJ'
]

CRYPTO_METADATA = {
    'BTC': {'price': 95000.0, 'volatility': 0.04, 'sector': 'store_of_value'},
    'ETH': {'price': 3400.0, 'volatility': 0.05, 'sector': 'smart_contracts'},
    'XRP': {'price': 2.30, 'volatility': 0.06, 'sector': 'payments'},
    'SOL': {'price': 190.0, 'volatility': 0.07, 'sector': 'smart_contracts'},
    'DOGE': {'price': 0.32, 'volatility': 0.10, 'sector': 'meme'},
    'ADA': {'price': 0.90, 'volatility': 0.06, 'sector': 'smart_contracts'},
    'AVAX': {'price': 38.0, 'volatility': 0.08, 'sector': 'smart_contracts'},
    'SHIB': {'price': 0.000022, 'volatility': 0.12, 'sector': 'meme'},
    'DOT': {'price': 7.20, 'volatility': 0.07, 'sector': 'infrastructure'},
    'LINK': {'price': 22.0, 'volatility': 0.06, 'sector': 'oracle'},
    'MATIC': {'price': 0.48, 'volatility': 0.08, 'sector': 'layer2'},
    'UNI': {'price': 13.50, 'volatility': 0.07, 'sector': 'defi'},
    'ATOM': {'price': 6.80, 'volatility': 0.08, 'sector': 'infrastructure'},
    'LTC': {'price': 105.0, 'volatility': 0.05, 'sector': 'payments'},
    'XLM': {'price': 0.42, 'volatility': 0.07, 'sector': 'payments'},
    'NEAR': {'price': 5.20, 'volatility': 0.09, 'sector': 'smart_contracts'},
    'APT': {'price': 9.50, 'volatility': 0.10, 'sector': 'smart_contracts'},
    'ARB': {'price': 0.75, 'volatility': 0.09, 'sector': 'layer2'},
    'OP': {'price': 1.80, 'volatility': 0.09, 'sector': 'layer2'},
    'INJ': {'price': 22.0, 'volatility': 0.10, 'sector': 'defi'},
}


@dataclass
class CryptoPosition:
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float = 0.0
    leverage: float = 1.0
    margin_reserved: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    ai_signal: str = ""
    ai_confidence: float = 0.0
    status: str = "open"
    opened_at: str = ""
    closed_at: str = ""
    close_reason: str = ""


@dataclass
class CryptoTradingSession:
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
    positions: List[CryptoPosition] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    improvements: List[Dict] = field(default_factory=list)


@dataclass
class CryptoStrategyConfig:
    version: int = 1
    position_size_percent: float = 10.0
    max_positions: int = 10
    min_confidence: float = 45.0
    stop_loss_percent: float = 8.0
    take_profit_percent: float = 15.0
    max_portfolio_exposure: float = 70.0
    max_leverage: float = 3.0
    allowed_symbols: List[str] = field(default_factory=lambda: TOP_CRYPTOS.copy())


class SyntheticCryptoMarket:
    """Generate realistic synthetic crypto market data"""
    
    def __init__(self, seed: int = None):
        if seed:
            np.random.seed(seed)
        self.prices = {s: m['price'] for s, m in CRYPTO_METADATA.items()}
        self.price_history = {s: [m['price']] for s, m in CRYPTO_METADATA.items()}
        self.cycle = 0
        self.market_trend = 0.0
    
    def advance_cycle(self):
        """Simulate price movements using Geometric Brownian Motion"""
        self.cycle += 1
        
        self.market_trend = np.clip(
            self.market_trend + np.random.normal(0, 0.01),
            -0.03, 0.03
        )
        
        for symbol, meta in CRYPTO_METADATA.items():
            vol = meta['volatility']
            drift = self.market_trend + np.random.normal(0.0001, vol)
            self.prices[symbol] *= (1 + drift)
            self.prices[symbol] = max(0.0000001, self.prices[symbol])
            
            self.price_history[symbol].append(self.prices[symbol])
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)
    
    def get_price_history(self, symbol: str, periods: int = 30) -> List[float]:
        history = self.price_history.get(symbol, [100.0])
        return history[-periods:] if len(history) >= periods else history
    
    def get_market_data(self, symbol: str) -> Dict:
        price = self.get_price(symbol)
        history = self.get_price_history(symbol, 30)
        
        if len(history) >= 2:
            pct_change_24h = ((price - history[-2]) / history[-2]) * 100 if history[-2] > 0 else 0
        else:
            pct_change_24h = 0
        
        if len(history) >= 7:
            pct_change_7d = ((price - history[-7]) / history[-7]) * 100 if history[-7] > 0 else 0
        else:
            pct_change_7d = 0
        
        meta = CRYPTO_METADATA.get(symbol, {})
        
        return {
            'symbol': symbol,
            'price': price,
            'pct_change_24h': pct_change_24h,
            'pct_change_7d': pct_change_7d,
            'volatility': meta.get('volatility', 0.05),
            'sector': meta.get('sector', 'unknown'),
            'volume_24h': price * np.random.exponential(1000000),
            'market_cap': price * np.random.exponential(10000000000)
        }


class CryptoPaperTradingEngine:
    """Crypto Paper Trading Engine with aggressive settings"""
    
    CACHE_DIR = "data/crypto_cache"
    
    def __init__(self, starting_balance: float = 100000.0,
                 target_balance: float = 200000.0,
                 max_drawdown: float = 30.0,
                 synthetic_mode: bool = True):
        self.synthetic_mode = synthetic_mode
        self.synthetic_market = SyntheticCryptoMarket() if synthetic_mode else None
        self.strategy = CryptoStrategyConfig()
        self.session: Optional[CryptoTradingSession] = None
        self.starting_balance = starting_balance
        self.target_balance = target_balance
        self.max_drawdown = max_drawdown
        self.is_running = False
        os.makedirs(self.CACHE_DIR, exist_ok=True)
    
    def start_session(self) -> CryptoTradingSession:
        session_id = str(uuid.uuid4())
        self.session = CryptoTradingSession(
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
        print(f"Started crypto paper trading session {session_id}")
        print(f"Starting balance: ${self.starting_balance:,.2f}")
        print(f"Target: ${self.target_balance:,.2f}")
        print(f"Max drawdown: {self.max_drawdown}%")
        return self.session
    
    def get_ai_signals(self) -> List[Dict]:
        """Generate AI trading signals based on technical analysis"""
        signals = []
        
        for symbol in self.strategy.allowed_symbols:
            try:
                if self.synthetic_mode and self.synthetic_market:
                    data = self.synthetic_market.get_market_data(symbol)
                    history = self.synthetic_market.get_price_history(symbol, 30)
                    
                    if len(history) < 14:
                        continue
                    
                    current_price = data['price']
                    pct_change = data['pct_change_24h']
                    pct_change_7d = data['pct_change_7d']
                    
                    rsi = self._calculate_rsi(np.array(history), 14)
                    sma_20 = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
                    
                    volatility = data['volatility']
                else:
                    continue
                
                signal = None
                confidence = 0
                
                if rsi < 30 and current_price > sma_20 * 0.95:
                    signal = 'LONG'
                    confidence = min(85, 55 + (30 - rsi) * 1.5)
                elif rsi > 70 and current_price < sma_20 * 1.05:
                    signal = 'SHORT'
                    confidence = min(85, 55 + (rsi - 70) * 1.5)
                elif pct_change > 5 and rsi < 65:
                    signal = 'LONG'
                    confidence = min(80, 50 + pct_change * 2)
                elif pct_change < -5 and rsi > 35:
                    signal = 'SHORT'
                    confidence = min(80, 50 + abs(pct_change) * 2)
                elif pct_change_7d > 15 and rsi < 60:
                    signal = 'LONG'
                    confidence = min(75, 45 + pct_change_7d)
                elif pct_change_7d < -15 and rsi > 40:
                    signal = 'SHORT'
                    confidence = min(75, 45 + abs(pct_change_7d))
                elif self.synthetic_mode and np.random.random() < 0.25:
                    signal = 'LONG' if np.random.random() < 0.55 else 'SHORT'
                    confidence = 45 + np.random.random() * 25
                
                if signal and confidence >= self.strategy.min_confidence:
                    leverage = min(self.strategy.max_leverage, 1 + (volatility * 10))
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'current_price': current_price,
                        'rsi': rsi,
                        'sma_20': sma_20,
                        'pct_change_24h': pct_change,
                        'pct_change_7d': pct_change_7d,
                        'volatility': volatility,
                        'suggested_leverage': leverage
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
        
        open_positions = [p for p in self.session.positions if p.status == 'open']
        
        if len(open_positions) >= self.strategy.max_positions:
            return {'success': False, 'reason': 'Max positions reached'}
        
        reserved_margin = sum(p.margin_reserved for p in open_positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
        current_equity = self.session.current_balance + reserved_margin + unrealized_pnl
        
        current_exposure = sum(p.quantity * p.entry_price * p.leverage for p in open_positions)
        exposure_pct = (current_exposure / current_equity) * 100 if current_equity > 0 else 100
        
        if exposure_pct >= self.strategy.max_portfolio_exposure:
            return {'success': False, 'reason': f'Portfolio exposure at {exposure_pct:.0f}%'}
        
        position_value = self.session.current_balance * (self.strategy.position_size_percent / 100)
        leverage = min(signal.get('suggested_leverage', 1.0), self.strategy.max_leverage)
        buying_power = position_value * leverage
        
        quantity = buying_power / signal['current_price']
        margin_required = position_value
        fees = margin_required * 0.001
        
        if margin_required + fees > self.session.current_balance:
            margin_required = self.session.current_balance * 0.9
            quantity = (margin_required * leverage) / signal['current_price']
            fees = margin_required * 0.001
            
            if margin_required + fees > self.session.current_balance:
                return {'success': False, 'reason': 'Insufficient funds'}
        
        position_id = str(uuid.uuid4())
        position = CryptoPosition(
            position_id=position_id,
            symbol=signal['symbol'],
            side='long' if signal['signal'] == 'LONG' else 'short',
            quantity=quantity,
            entry_price=signal['current_price'],
            current_price=signal['current_price'],
            leverage=leverage,
            margin_reserved=margin_required,
            ai_signal=signal['signal'],
            ai_confidence=signal['confidence'],
            opened_at=datetime.now().isoformat()
        )
        
        self.session.positions.append(position)
        self.session.current_balance -= (margin_required + fees)
        self.session.total_trades += 1
        
        trade_record = {
            'trade_id': str(uuid.uuid4()),
            'position_id': position_id,
            'symbol': signal['symbol'],
            'action': 'OPEN',
            'side': position.side,
            'quantity': quantity,
            'price': signal['current_price'],
            'leverage': leverage,
            'margin': margin_required,
            'fees': fees,
            'ai_signal': signal['signal'],
            'ai_confidence': signal['confidence'],
            'executed_at': datetime.now().isoformat()
        }
        self.session.trade_history.append(trade_record)
        
        print(f"Opened {position.side.upper()}: {quantity:.4f} {signal['symbol']} @ ${signal['current_price']:,.2f} ({leverage:.1f}x)")
        
        return {'success': True, 'position': asdict(position), 'trade': trade_record}
    
    def update_positions(self) -> Dict[str, Any]:
        if not self.session:
            return {}
        
        total_unrealized = 0.0
        reserved_margin = 0.0
        
        for position in self.session.positions:
            if position.status != 'open':
                continue
            
            if self.synthetic_mode and self.synthetic_market:
                position.current_price = self.synthetic_market.get_price(position.symbol)
            
            if position.side == 'long':
                price_change = (position.current_price - position.entry_price) / position.entry_price
            else:
                price_change = (position.entry_price - position.current_price) / position.entry_price
            
            position_notional = position.quantity * position.entry_price
            position.unrealized_pnl = position_notional * price_change * position.leverage
            
            total_unrealized += position.unrealized_pnl
            reserved_margin += position.margin_reserved
            
            pnl_percent = (position.unrealized_pnl / position.margin_reserved) * 100 if position.margin_reserved > 0 else 0
            
            if pnl_percent <= -self.strategy.stop_loss_percent:
                self._close_position(position, 'stop_loss')
            elif pnl_percent >= self.strategy.take_profit_percent:
                self._close_position(position, 'take_profit')
        
        self.session.total_unrealized_pnl = total_unrealized
        
        portfolio_value = self.session.current_balance + reserved_margin + total_unrealized
        
        if portfolio_value > self.session.peak_balance:
            self.session.peak_balance = portfolio_value
        
        drawdown = ((self.session.peak_balance - portfolio_value) / self.session.peak_balance) * 100
        self.session.current_drawdown = drawdown
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.session.current_balance,
            'reserved_margin': reserved_margin,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': self.session.total_realized_pnl,
            'drawdown': drawdown,
            'peak_balance': self.session.peak_balance
        }
    
    def _close_position(self, position: CryptoPosition, reason: str):
        if position.status != 'open':
            return
        
        position_notional = position.quantity * position.entry_price
        margin = position_notional / position.leverage
        
        if position.side == 'long':
            price_change = (position.current_price - position.entry_price) / position.entry_price
        else:
            price_change = (position.entry_price - position.current_price) / position.entry_price
        
        realized_pnl = position_notional * price_change * position.leverage
        fees = abs(realized_pnl) * 0.001
        realized_pnl -= fees
        
        position.realized_pnl = realized_pnl
        position.status = 'closed'
        position.closed_at = datetime.now().isoformat()
        position.close_reason = reason
        
        self.session.current_balance += margin + realized_pnl
        self.session.total_realized_pnl += realized_pnl
        
        if realized_pnl >= 0:
            self.session.winning_trades += 1
        else:
            self.session.losing_trades += 1
        
        print(f"Closed {position.side.upper()} ({reason}): {position.symbol}")
        print(f"  P&L: ${realized_pnl:,.2f}")
    
    def check_session_status(self) -> Dict[str, Any]:
        if not self.session:
            return {'status': 'no_session'}
        
        position_update = self.update_positions()
        portfolio_value = position_update.get('portfolio_value', self.session.current_balance)
        drawdown = position_update.get('drawdown', 0)
        
        if portfolio_value >= self.session.target_balance:
            self._end_session('target_reached')
            return {
                'status': 'target_reached',
                'portfolio_value': portfolio_value,
                'drawdown': drawdown
            }
        
        if drawdown >= self.session.max_drawdown_limit:
            analysis = self._analyze_session()
            improvements = self._generate_improvements(analysis)
            self._end_session('drawdown_exceeded')
            return {
                'status': 'drawdown_exceeded',
                'portfolio_value': portfolio_value,
                'drawdown': drawdown,
                'analysis': analysis,
                'improvements': improvements
            }
        
        return {
            'status': 'active',
            'portfolio_value': portfolio_value,
            'drawdown': drawdown
        }
    
    def run_trading_cycle(self) -> Dict[str, Any]:
        if not self.session or self.session.status != 'active':
            return {'error': 'No active session'}
        
        if self.synthetic_mode and self.synthetic_market:
            self.synthetic_market.advance_cycle()
        
        status = self.check_session_status()
        if status['status'] != 'active':
            return {'status': status}
        
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
        
        final_status = self.check_session_status()
        
        return {
            'status': final_status,
            'trades_executed': len(executed),
            'signals_found': len(signals),
            'executed_trades': executed
        }
    
    def _analyze_session(self) -> Dict[str, Any]:
        if not self.session:
            return {}
        
        closed_positions = [p for p in self.session.positions if p.status == 'closed']
        
        total_pnl = sum(p.realized_pnl for p in closed_positions)
        wins = [p for p in closed_positions if p.realized_pnl >= 0]
        losses = [p for p in closed_positions if p.realized_pnl < 0]
        
        win_rate = (len(wins) / len(closed_positions) * 100) if closed_positions else 0
        avg_win = sum(p.realized_pnl for p in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(p.realized_pnl for p in losses) / len(losses)) if losses else 0
        
        total_wins = sum(p.realized_pnl for p in wins)
        total_losses = abs(sum(p.realized_pnl for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        symbol_performance = {}
        for p in closed_positions:
            if p.symbol not in symbol_performance:
                symbol_performance[p.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0, 'trades': 0}
            symbol_performance[p.symbol]['trades'] += 1
            symbol_performance[p.symbol]['pnl'] += p.realized_pnl
            if p.realized_pnl >= 0:
                symbol_performance[p.symbol]['wins'] += 1
            else:
                symbol_performance[p.symbol]['losses'] += 1
        
        losing_symbols = [s for s, data in symbol_performance.items() 
                         if data['trades'] >= 2 and data['pnl'] < 0]
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(closed_positions),
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
            improvements['new_parameters']['position_size_percent'] = max(3, self.strategy.position_size_percent * 0.8)
            
            improvements['strategy_changes'].append('Reduce leverage')
            improvements['new_parameters']['max_leverage'] = max(1.0, self.strategy.max_leverage * 0.8)
        
        if analysis['win_rate'] < 40:
            improvements['strategy_changes'].append('Increase confidence threshold')
            improvements['new_parameters']['min_confidence'] = min(70, self.strategy.min_confidence + 5)
        
        if analysis['profit_factor'] < 0.8:
            improvements['strategy_changes'].append('Tighten stop loss')
            improvements['new_parameters']['stop_loss_percent'] = max(5, self.strategy.stop_loss_percent - 2)
        
        if analysis['avg_loss'] > analysis['avg_win'] * 1.5 and analysis['total_pnl'] < 0:
            improvements['strategy_changes'].append('Lower portfolio exposure')
            improvements['new_parameters']['max_portfolio_exposure'] = max(40, self.strategy.max_portfolio_exposure * 0.85)
        
        for symbol in analysis['losing_symbols']:
            data = analysis['symbol_performance'][symbol]
            if data['trades'] >= 3 and data['wins'] / data['trades'] < 0.3:
                improvements['symbols_to_exclude'].append(symbol)
                improvements['strategy_changes'].append(f'Exclude {symbol}')
        
        return improvements
    
    def apply_improvements(self, improvements: Dict) -> CryptoStrategyConfig:
        self.strategy.version = improvements.get('version', self.strategy.version + 1)
        
        for param, value in improvements.get('new_parameters', {}).items():
            if hasattr(self.strategy, param):
                setattr(self.strategy, param, value)
        
        for symbol in improvements.get('symbols_to_exclude', []):
            if symbol in self.strategy.allowed_symbols:
                self.strategy.allowed_symbols.remove(symbol)
        
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
    
    def get_session_summary(self) -> Dict[str, Any]:
        if not self.session:
            return {}
        
        open_positions = [p for p in self.session.positions if p.status == 'open']
        reserved_margin = sum(p.margin_reserved for p in open_positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
        portfolio_value = self.session.current_balance + reserved_margin + unrealized_pnl
        return_pct = ((portfolio_value - self.session.starting_balance) / self.session.starting_balance) * 100
        
        return {
            'session_id': self.session.session_id,
            'portfolio_value': portfolio_value,
            'starting_balance': self.session.starting_balance,
            'return_pct': return_pct,
            'total_trades': self.session.total_trades,
            'winning_trades': self.session.winning_trades,
            'losing_trades': self.session.losing_trades,
            'win_rate': (self.session.winning_trades / self.session.total_trades * 100) if self.session.total_trades > 0 else 0,
            'total_realized_pnl': self.session.total_realized_pnl,
            'max_drawdown': self.session.current_drawdown,
            'strategy_version': self.strategy.version,
            'status': self.session.status,
            'ended_at': self.session.ended_at,
            'end_reason': self.session.end_reason
        }
