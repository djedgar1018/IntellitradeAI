"""Parse trade messages from Discord to extract trade information."""

import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ParsedTrade:
    """Represents a parsed trade from a Discord message."""
    message_id: str
    author: str
    timestamp: datetime
    symbol: str
    asset_type: str
    action: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    strike_price: Optional[float] = None
    expiration: Optional[str] = None
    option_type: Optional[str] = None
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    outcome: Optional[str] = None
    raw_message: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return d


class TradeMessageParser:
    """Parse Discord messages to extract trade information."""
    
    CRYPTO_SYMBOLS = [
        'BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'AVAX', 'SHIB', 'DOT', 'LINK',
        'MATIC', 'UNI', 'ATOM', 'LTC', 'XLM', 'NEAR', 'APT', 'ARB', 'OP', 'INJ',
        'FET', 'RNDR', 'TAO', 'PEPE', 'BONK', 'WIF', 'FLOKI', 'MEME', 'BRETT'
    ]
    
    STOCK_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC',
        'NFLX', 'DIS', 'BA', 'JPM', 'BAC', 'WMT', 'HD', 'V', 'MA', 'PFE', 'JNJ',
        'TSM', 'MU', 'WDC', 'PLTR', 'HOOD', 'GEV', 'LLY', 'SPY', 'QQQ', 'IWM'
    ]
    
    ACTION_PATTERNS = {
        'buy': [r'\bbuy\b', r'\bbought\b', r'\blong\b', r'\bentry\b', r'\bentered\b', r'\bopening\b'],
        'sell': [r'\bsell\b', r'\bsold\b', r'\bshort\b', r'\bexit\b', r'\bclosed\b', r'\bclosing\b'],
        'call': [r'\bcall[s]?\b', r'\bc\s*\d'],
        'put': [r'\bput[s]?\b', r'\bp\s*\d']
    }
    
    def __init__(self):
        self.price_pattern = re.compile(r'\$?\d+\.?\d*')
        self.percentage_pattern = re.compile(r'[+-]?\d+\.?\d*\s*%')
        self.strike_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:c|p|call|put)', re.IGNORECASE)
        self.expiry_pattern = re.compile(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)|(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))', re.IGNORECASE)
    
    def parse_message(self, message: Dict) -> Optional[ParsedTrade]:
        """Parse a single Discord message for trade information."""
        content = message.get('content', '')
        if not content or len(content) < 5:
            return None
        
        content_lower = content.lower()
        
        symbol = self._extract_symbol(content)
        if not symbol:
            return None
        
        asset_type = self._determine_asset_type(symbol, content_lower)
        action = self._extract_action(content_lower, asset_type)
        
        if not action:
            return None
        
        author = message.get('author', {}).get('username', 'Unknown')
        timestamp = self._parse_timestamp(message.get('timestamp', ''))
        
        trade = ParsedTrade(
            message_id=message.get('id', ''),
            author=author,
            timestamp=timestamp,
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            raw_message=content
        )
        
        prices = self._extract_prices(content)
        if prices:
            if action in ['buy', 'call', 'put']:
                trade.entry_price = prices[0]
                if len(prices) > 1:
                    trade.exit_price = prices[-1]
            else:
                trade.exit_price = prices[0]
                if len(prices) > 1:
                    trade.entry_price = prices[-1]
        
        if asset_type == 'option':
            strike = self._extract_strike_price(content)
            if strike:
                trade.strike_price = strike
            
            expiry = self._extract_expiration(content)
            if expiry:
                trade.expiration = expiry
            
            trade.option_type = 'call' if action == 'call' or 'call' in content_lower else 'put'
        
        pnl = self._extract_profit_loss(content)
        if pnl:
            trade.profit_loss = pnl.get('amount')
            trade.profit_loss_pct = pnl.get('percentage')
            if pnl.get('amount', 0) > 0 or pnl.get('percentage', 0) > 0:
                trade.outcome = 'win'
            elif pnl.get('amount', 0) < 0 or pnl.get('percentage', 0) < 0:
                trade.outcome = 'loss'
        
        reasoning = self._extract_reasoning(content)
        if reasoning:
            trade.reasoning = reasoning
        
        return trade
    
    def _extract_symbol(self, content: str) -> Optional[str]:
        """Extract trading symbol from message content."""
        content_upper = content.upper()
        
        for symbol in self.STOCK_SYMBOLS + self.CRYPTO_SYMBOLS:
            pattern = rf'\b{symbol}\b'
            if re.search(pattern, content_upper):
                return symbol
        
        ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        match = ticker_pattern.search(content_upper)
        if match:
            return match.group(1)
        
        return None
    
    def _determine_asset_type(self, symbol: str, content_lower: str) -> str:
        """Determine if trade is crypto, stock, or option."""
        if symbol in self.CRYPTO_SYMBOLS:
            return 'crypto'
        
        option_indicators = ['call', 'put', 'strike', 'expir', 'contract', '/c', '/p', 'dte']
        if any(ind in content_lower for ind in option_indicators):
            return 'option'
        
        return 'stock'
    
    def _extract_action(self, content_lower: str, asset_type: str) -> Optional[str]:
        """Extract trade action from message."""
        if asset_type == 'option':
            for pattern in self.ACTION_PATTERNS['call']:
                if re.search(pattern, content_lower):
                    return 'call'
            for pattern in self.ACTION_PATTERNS['put']:
                if re.search(pattern, content_lower):
                    return 'put'
        
        for pattern in self.ACTION_PATTERNS['buy']:
            if re.search(pattern, content_lower):
                return 'buy'
        
        for pattern in self.ACTION_PATTERNS['sell']:
            if re.search(pattern, content_lower):
                return 'sell'
        
        return None
    
    def _extract_prices(self, content: str) -> List[float]:
        """Extract price values from message."""
        prices = []
        
        price_patterns = [
            r'\$(\d+\.?\d*)',
            r'@\s*(\d+\.?\d*)',
            r'price[:\s]+(\d+\.?\d*)',
            r'entry[:\s]+\$?(\d+\.?\d*)',
            r'exit[:\s]+\$?(\d+\.?\d*)',
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match)
                    if 0.0001 < price < 1000000:
                        prices.append(price)
                except ValueError:
                    continue
        
        return sorted(set(prices))
    
    def _extract_strike_price(self, content: str) -> Optional[float]:
        """Extract option strike price."""
        match = self.strike_pattern.search(content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_expiration(self, content: str) -> Optional[str]:
        """Extract option expiration date."""
        match = self.expiry_pattern.search(content)
        if match:
            return match.group(0)
        return None
    
    def _extract_profit_loss(self, content: str) -> Optional[Dict]:
        """Extract profit/loss information."""
        pnl = {}
        
        pct_patterns = [
            r'([+-]?\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%\s*(gain|profit|up)',
            r'(\d+\.?\d*)\s*%\s*(loss|down)',
        ]
        
        for pattern in pct_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if 'loss' in content.lower() or 'down' in content.lower():
                        value = -abs(value)
                    elif 'gain' in content.lower() or 'profit' in content.lower() or 'up' in content.lower():
                        value = abs(value)
                    pnl['percentage'] = value
                    break
                except ValueError:
                    continue
        
        amount_patterns = [
            r'([+-]?\$?\d+(?:,?\d{3})*(?:\.\d{2})?)\s*(?:profit|gain|loss)',
            r'(?:profit|gain|loss|p/?l)[:\s]*([+-]?\$?\d+(?:,?\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace('$', '').replace(',', '')
                    value = float(amount_str)
                    if 'loss' in content.lower():
                        value = -abs(value)
                    pnl['amount'] = value
                    break
                except ValueError:
                    continue
        
        return pnl if pnl else None
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract trading reasoning or thesis from message."""
        reasoning_patterns = [
            r'because\s+(.+?)(?:\.|$)',
            r'reason[:\s]+(.+?)(?:\.|$)',
            r'thesis[:\s]+(.+?)(?:\.|$)',
            r'expecting\s+(.+?)(?:\.|$)',
            r'bullish\s+(?:on\s+)?(.+?)(?:\.|$)',
            r'bearish\s+(?:on\s+)?(.+?)(?:\.|$)',
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 10:
                    return reasoning[:500]
        
        return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Discord timestamp string."""
        try:
            if 'Z' in timestamp_str:
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            return datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
        except Exception:
            return datetime.now()
    
    def parse_messages(self, messages: List[Dict]) -> List[ParsedTrade]:
        """Parse multiple Discord messages and return list of trades."""
        trades = []
        for message in messages:
            trade = self.parse_message(message)
            if trade:
                trades.append(trade)
        return trades
    
    def get_trade_summary(self, trades: List[ParsedTrade]) -> Dict:
        """Generate summary statistics from parsed trades."""
        if not trades:
            return {'total_trades': 0}
        
        wins = [t for t in trades if t.outcome == 'win']
        losses = [t for t in trades if t.outcome == 'loss']
        
        symbols = {}
        for trade in trades:
            symbols[trade.symbol] = symbols.get(trade.symbol, 0) + 1
        
        asset_types = {}
        for trade in trades:
            asset_types[trade.asset_type] = asset_types.get(trade.asset_type, 0) + 1
        
        actions = {}
        for trade in trades:
            actions[trade.action] = actions.get(trade.action, 0) + 1
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'symbols_traded': dict(sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:20]),
            'asset_types': asset_types,
            'actions': actions,
            'date_range': {
                'start': min(t.timestamp for t in trades).isoformat() if trades else None,
                'end': max(t.timestamp for t in trades).isoformat() if trades else None
            }
        }
