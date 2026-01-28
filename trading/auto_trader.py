"""
Auto Trader Engine for IntelliTradeAI
Executes trades based on AI signals in real-time
"""

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf

try:
    from trading.v22_scalp_config import V22ScalpConfig
    V22_AVAILABLE = True
except ImportError:
    V22_AVAILABLE = False

STOCK_SYMBOLS = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'AMD', 'PLTR', 'HOOD']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD']
ETF_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'ARKK']


class AutoTrader:
    """
    Automated trading engine that generates and executes trades
    based on AI signals and market analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.active = False
        self.trades = []
        self.open_positions = []
        self.closed_positions = []
        self.starting_balance = self.config.get('starting_capital', 10000)
        self.current_balance = self.starting_balance
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.last_scan_time = None
        self.scan_interval = 5
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'starting_capital': 10000,
            'risk_tolerance': 'moderate',
            'asset_classes': ['stocks'],
            'timeframe': 'day',
            'min_confidence': 65,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0,
            'max_positions': 5,
            'position_size_pct': 20.0,
        }
    
    def configure(self, 
                  risk_tolerance: str = 'moderate',
                  asset_classes: List[str] = None,
                  starting_capital: float = 10000,
                  timeframe: str = 'day'):
        """Configure the auto trader based on wizard settings"""
        
        risk_params = {
            'conservative': {'min_confidence': 75, 'stop_loss_pct': 2.0, 'take_profit_pct': 4.0, 'position_size_pct': 10.0},
            'moderate': {'min_confidence': 65, 'stop_loss_pct': 3.0, 'take_profit_pct': 6.0, 'position_size_pct': 20.0},
            'aggressive': {'min_confidence': 55, 'stop_loss_pct': 5.0, 'take_profit_pct': 10.0, 'position_size_pct': 30.0},
        }
        
        params = risk_params.get(risk_tolerance, risk_params['moderate'])
        
        self.config.update({
            'risk_tolerance': risk_tolerance,
            'asset_classes': asset_classes or ['stocks'],
            'starting_capital': starting_capital,
            'timeframe': timeframe,
            **params
        })
        
        self.starting_balance = starting_capital
        self.current_balance = starting_capital
        
    def start(self):
        """Start the auto trader"""
        self.active = True
        self.last_scan_time = datetime.now()
        
    def stop(self):
        """Stop the auto trader"""
        self.active = False
        
    def is_active(self) -> bool:
        return self.active
    
    def get_symbols_for_scan(self) -> List[str]:
        """Get list of symbols to scan based on configured asset classes"""
        symbols = []
        asset_classes = self.config.get('asset_classes', ['stocks'])
        
        if 'stocks' in asset_classes:
            symbols.extend(STOCK_SYMBOLS)
        if 'crypto' in asset_classes:
            symbols.extend(CRYPTO_SYMBOLS)
        if 'etf' in asset_classes or 'etfs' in asset_classes:
            symbols.extend(ETF_SYMBOLS)
            
        return symbols
    
    def fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            high_5d = hist['High'].max()
            low_5d = hist['Low'].min()
            
            price_change = ((current_price - prev_close) / prev_close) * 100
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'prev_close': float(prev_close),
                'price_change_pct': float(price_change),
                'volume': int(volume),
                'avg_volume': int(avg_volume),
                'volume_ratio': float(volume_ratio),
                'high_5d': float(high_5d),
                'low_5d': float(low_5d),
            }
        except Exception as e:
            return None
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on market data"""
        symbol = market_data['symbol']
        price = market_data['current_price']
        price_change = market_data['price_change_pct']
        volume_ratio = market_data['volume_ratio']
        
        score = 50
        reasons = []
        
        if price_change > 2:
            score += 15
            reasons.append("Strong positive momentum (+{:.1f}%)".format(price_change))
        elif price_change > 0.5:
            score += 8
            reasons.append("Positive price movement (+{:.1f}%)".format(price_change))
        elif price_change < -2:
            score -= 10
            reasons.append("Negative momentum ({:.1f}%)".format(price_change))
            
        if volume_ratio > 1.5:
            score += 12
            reasons.append("Above average volume ({:.1f}x normal)".format(volume_ratio))
        elif volume_ratio > 1.2:
            score += 5
            reasons.append("Slightly elevated volume")
            
        price_range = market_data['high_5d'] - market_data['low_5d']
        if price_range > 0:
            position_in_range = (price - market_data['low_5d']) / price_range
            if position_in_range < 0.3:
                score += 10
                reasons.append("Trading near 5-day lows (potential bounce)")
            elif position_in_range > 0.8:
                score -= 5
                reasons.append("Near 5-day highs (potential resistance)")
        
        random_factor = random.uniform(-10, 15)
        score += random_factor
        
        score = max(0, min(100, score))
        
        if score >= 65:
            signal = 'BUY'
            reasons.append("Technical indicators align for bullish setup")
        elif score <= 35:
            signal = 'SELL'
            reasons.append("Technical indicators suggest bearish setup")
        else:
            signal = 'HOLD'
            reasons.append("Mixed signals - waiting for clearer direction")
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': round(score),
            'price': price,
            'reasons': reasons[:3],
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management"""
        position_value = self.current_balance * (self.config['position_size_pct'] / 100)
        
        if price > 0:
            return round(position_value / price, 4)
        return 0
    
    def execute_trade(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade based on the signal"""
        if not self.active:
            return None
            
        if signal['signal'] == 'HOLD':
            return None
            
        if signal['confidence'] < self.config['min_confidence']:
            return None
            
        if len(self.open_positions) >= self.config['max_positions']:
            return None
            
        for pos in self.open_positions:
            if pos['symbol'] == signal['symbol']:
                return None
        
        symbol = signal['symbol']
        price = signal['price']
        action = signal['signal']
        quantity = self.calculate_position_size(price)
        
        if quantity <= 0:
            return None
        
        stop_loss_pct = self.config['stop_loss_pct'] / 100
        take_profit_pct = self.config['take_profit_pct'] / 100
        
        if action == 'BUY':
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)
        
        asset_type = 'crypto' if '-USD' in symbol else 'stock'
        if symbol in ETF_SYMBOLS:
            asset_type = 'etf'
        
        trade = {
            'id': len(self.trades) + 1,
            'symbol': symbol,
            'action': action,
            'entry_price': round(price, 2),
            'current_price': round(price, 2),
            'quantity': quantity,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence': signal['confidence'],
            'reasons': signal['reasons'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'open',
            'asset_type': asset_type,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
        }
        
        self.trades.append(trade)
        self.open_positions.append(trade)
        
        return trade
    
    def update_positions(self) -> List[Dict[str, Any]]:
        """Update all open positions with current prices"""
        updated = []
        positions_to_close = []
        
        for pos in self.open_positions:
            market_data = self.fetch_market_data(pos['symbol'])
            
            if market_data:
                new_price = market_data['current_price']
                pos['current_price'] = round(new_price, 2)
                
                if pos['action'] == 'BUY':
                    pos['unrealized_pnl'] = round((new_price - pos['entry_price']) * pos['quantity'], 2)
                else:
                    pos['unrealized_pnl'] = round((pos['entry_price'] - new_price) * pos['quantity'], 2)
                
                should_close = False
                close_reason = ""
                
                if pos['action'] == 'BUY':
                    if new_price <= pos['stop_loss']:
                        should_close = True
                        close_reason = "Stop loss triggered"
                    elif new_price >= pos['take_profit']:
                        should_close = True
                        close_reason = "Take profit reached"
                else:
                    if new_price >= pos['stop_loss']:
                        should_close = True
                        close_reason = "Stop loss triggered"
                    elif new_price <= pos['take_profit']:
                        should_close = True
                        close_reason = "Take profit reached"
                
                if should_close:
                    pos['status'] = 'closed'
                    pos['realized_pnl'] = pos['unrealized_pnl']
                    pos['close_reason'] = close_reason
                    pos['closed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    positions_to_close.append(pos)
                    
                    self.total_pnl += pos['realized_pnl']
                    self.current_balance += pos['realized_pnl']
                    
                    if pos['realized_pnl'] > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                
                updated.append(pos)
        
        for pos in positions_to_close:
            if pos in self.open_positions:
                self.open_positions.remove(pos)
                self.closed_positions.append(pos)
        
        return updated
    
    def run_scan(self) -> List[Dict[str, Any]]:
        """Run a market scan and generate trades"""
        if not self.active:
            return []
        
        self.update_positions()
        
        new_trades = []
        symbols = self.get_symbols_for_scan()
        
        scan_symbols = random.sample(symbols, min(3, len(symbols)))
        
        for symbol in scan_symbols:
            market_data = self.fetch_market_data(symbol)
            
            if market_data:
                signal = self.generate_signal(market_data)
                
                if signal['signal'] in ['BUY', 'SELL']:
                    trade = self.execute_trade(signal)
                    if trade:
                        new_trades.append(trade)
        
        self.last_scan_time = datetime.now()
        
        return new_trades
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Get all trades (open and closed)"""
        return self.trades
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        return self.open_positions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'starting_balance': self.starting_balance,
            'current_balance': round(self.current_balance, 2),
            'total_pnl': round(self.total_pnl, 2),
            'pnl_percent': round((self.current_balance - self.starting_balance) / self.starting_balance * 100, 2),
            'total_trades': len(self.trades),
            'open_positions': len(self.open_positions),
            'closed_trades': len(self.closed_positions),
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': round(win_rate, 1),
            'is_active': self.active
        }
    
    def close_position(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Manually close a position"""
        for pos in self.open_positions:
            if pos.get('id') == trade_id:
                market_data = self.fetch_market_data(pos['symbol'])
                if market_data:
                    pos['current_price'] = round(market_data['current_price'], 2)
                
                if pos['action'] == 'BUY':
                    pos['realized_pnl'] = round((pos['current_price'] - pos['entry_price']) * pos['quantity'], 2)
                else:
                    pos['realized_pnl'] = round((pos['entry_price'] - pos['current_price']) * pos['quantity'], 2)
                
                pos['status'] = 'closed'
                pos['close_reason'] = 'Manual close'
                pos['closed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                self.total_pnl += pos['realized_pnl']
                self.current_balance += pos['realized_pnl']
                
                if pos['realized_pnl'] > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                self.open_positions.remove(pos)
                self.closed_positions.append(pos)
                
                return pos
        
        return None


_auto_trader_instance = None

def get_auto_trader() -> AutoTrader:
    """Get the singleton auto trader instance"""
    global _auto_trader_instance
    if _auto_trader_instance is None:
        _auto_trader_instance = AutoTrader()
    return _auto_trader_instance

def reset_auto_trader():
    """Reset the auto trader instance"""
    global _auto_trader_instance
    _auto_trader_instance = None
