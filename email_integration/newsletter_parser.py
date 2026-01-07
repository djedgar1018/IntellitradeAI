"""Parser for extracting trading signals from newsletter emails."""

import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EmailSignal:
    """Represents a trading signal extracted from an email."""
    symbol: str
    signal_type: str  # buy, sell, bullish, bearish, hold, watch
    source: str  # tldr, barchart, investing, webull
    subject: str
    price_target: Optional[float] = None
    current_price: Optional[float] = None
    confidence: float = 0.5
    reasoning: str = ""
    asset_type: str = "stock"  # stock, crypto, etf, option
    timeframe: str = "swing"  # day, swing, long-term
    email_date: Optional[str] = None
    raw_text: str = ""


class NewsletterParser:
    """Parse trading signals from various newsletter sources."""
    
    CRYPTO_SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LINK']
    
    BULLISH_KEYWORDS = ['buy', 'bullish', 'upgrade', 'outperform', 'strong buy', 'accumulate', 
                        'overweight', 'positive', 'upside', 'breakout', 'momentum', 'surge',
                        'rally', 'gains', 'growth', 'beat', 'exceeded']
    
    BEARISH_KEYWORDS = ['sell', 'bearish', 'downgrade', 'underperform', 'underweight',
                        'negative', 'downside', 'breakdown', 'decline', 'drop', 'fall',
                        'miss', 'warning', 'risk', 'pullback']
    
    def __init__(self):
        self.signals: List[EmailSignal] = []
    
    def parse_email(self, email: Dict) -> List[EmailSignal]:
        """Parse a single email for trading signals."""
        sender = email.get('from', '').lower()
        subject = email.get('subject', '')
        body = email.get('body', '')
        date = email.get('date', '')
        
        signals = []
        
        if 'tldr' in sender:
            signals.extend(self._parse_tldr(email))
        elif 'barchart' in sender:
            signals.extend(self._parse_barchart(email))
        elif 'investing.com' in sender:
            signals.extend(self._parse_investing(email))
        elif 'webull' in sender:
            signals.extend(self._parse_webull(email))
        else:
            signals.extend(self._parse_generic(email))
        
        return signals
    
    def parse_emails(self, emails: List[Dict]) -> List[EmailSignal]:
        """Parse multiple emails for trading signals."""
        all_signals = []
        
        for email in emails:
            signals = self.parse_email(email)
            all_signals.extend(signals)
        
        self.signals = all_signals
        return all_signals
    
    def _parse_tldr(self, email: Dict) -> List[EmailSignal]:
        """Parse TLDR AI newsletter for trading signals."""
        signals = []
        body = email.get('body', '')
        subject = email.get('subject', '')
        date = email.get('date', '')
        
        stock_pattern = r'\b([A-Z]{1,5})\b[:\s]+(?:stock|shares?|price)[^\n]*?(\$[\d,.]+|\d+%)'
        matches = re.findall(stock_pattern, body)
        
        for match in matches:
            symbol = match[0]
            if len(symbol) >= 2 and symbol not in ['AI', 'THE', 'AND', 'FOR', 'CEO', 'IPO']:
                signal_type = self._detect_sentiment(body, symbol)
                
                signals.append(EmailSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    source='tldr',
                    subject=subject,
                    confidence=0.6,
                    reasoning=f"Mentioned in TLDR AI newsletter",
                    email_date=date,
                    raw_text=body[:500]
                ))
        
        ai_stocks = ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'AMD', 'TSLA', 'PLTR', 'SNOW', 'CRM']
        for symbol in ai_stocks:
            if symbol in body.upper() and not any(s.symbol == symbol for s in signals):
                signal_type = self._detect_sentiment(body, symbol)
                signals.append(EmailSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    source='tldr',
                    subject=subject,
                    confidence=0.5,
                    reasoning=f"AI-related stock mentioned in TLDR",
                    email_date=date
                ))
        
        return signals
    
    def _parse_barchart(self, email: Dict) -> List[EmailSignal]:
        """Parse Barchart newsletter for trading signals."""
        signals = []
        body = email.get('body', '')
        subject = email.get('subject', '')
        date = email.get('date', '')
        
        rating_pattern = r'([A-Z]{1,5})\s*[-–]\s*(Strong Buy|Buy|Hold|Sell|Strong Sell)'
        matches = re.findall(rating_pattern, body, re.IGNORECASE)
        
        for symbol, rating in matches:
            if len(symbol) >= 2:
                rating_lower = rating.lower()
                if 'strong buy' in rating_lower:
                    signal_type = 'strong_buy'
                    confidence = 0.9
                elif 'buy' in rating_lower:
                    signal_type = 'buy'
                    confidence = 0.7
                elif 'strong sell' in rating_lower:
                    signal_type = 'strong_sell'
                    confidence = 0.9
                elif 'sell' in rating_lower:
                    signal_type = 'sell'
                    confidence = 0.7
                else:
                    signal_type = 'hold'
                    confidence = 0.5
                
                signals.append(EmailSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    source='barchart',
                    subject=subject,
                    confidence=confidence,
                    reasoning=f"Barchart rating: {rating}",
                    email_date=date
                ))
        
        price_pattern = r'([A-Z]{1,5})\s*(?:target|price target|PT)[:\s]*\$?([\d,.]+)'
        price_matches = re.findall(price_pattern, body, re.IGNORECASE)
        
        for symbol, target in price_matches:
            if len(symbol) >= 2:
                try:
                    price_target = float(target.replace(',', ''))
                    existing = next((s for s in signals if s.symbol == symbol), None)
                    if existing:
                        existing.price_target = price_target
                    else:
                        signals.append(EmailSignal(
                            symbol=symbol,
                            signal_type='watch',
                            source='barchart',
                            subject=subject,
                            price_target=price_target,
                            confidence=0.6,
                            reasoning=f"Price target: ${price_target}",
                            email_date=date
                        ))
                except ValueError:
                    pass
        
        return signals
    
    def _parse_investing(self, email: Dict) -> List[EmailSignal]:
        """Parse Investing.com newsletter for trading signals."""
        signals = []
        body = email.get('body', '')
        subject = email.get('subject', '')
        date = email.get('date', '')
        
        patterns = [
            r'([A-Z]{1,5})\s+(?:rises?|gains?|jumps?|surges?|soars?)\s+(\d+(?:\.\d+)?%?)',
            r'([A-Z]{1,5})\s+(?:falls?|drops?|declines?|plunges?|tumbles?)\s+(\d+(?:\.\d+)?%?)',
            r'([A-Z]{1,5})\s+(?:up|down)\s+(\d+(?:\.\d+)?%)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for match in matches:
                symbol = match[0].upper()
                if len(symbol) >= 2 and symbol not in ['THE', 'AND', 'FOR']:
                    is_bullish = any(kw in pattern.lower() for kw in ['rises', 'gains', 'jumps', 'surges', 'up'])
                    signal_type = 'bullish' if is_bullish else 'bearish'
                    
                    signals.append(EmailSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        source='investing',
                        subject=subject,
                        confidence=0.6,
                        reasoning=f"Price movement reported by Investing.com",
                        email_date=date
                    ))
        
        return signals
    
    def _parse_webull(self, email: Dict) -> List[EmailSignal]:
        """Parse Webull newsletter for trading signals."""
        signals = []
        body = email.get('body', '')
        subject = email.get('subject', '')
        date = email.get('date', '')
        
        top_pattern = r'Top\s+(?:Gainers?|Losers?|Movers?)[\s\S]*?([A-Z]{1,5})(?:\s*[-–]\s*|\s+)([+-]?\d+(?:\.\d+)?%)'
        matches = re.findall(top_pattern, body, re.IGNORECASE)
        
        for symbol, change in matches:
            if len(symbol) >= 2:
                is_positive = '+' in change or (not '-' in change and float(change.replace('%', '')) > 0)
                signal_type = 'bullish' if is_positive else 'bearish'
                
                signals.append(EmailSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    source='webull',
                    subject=subject,
                    confidence=0.55,
                    reasoning=f"Webull top mover: {change}",
                    email_date=date
                ))
        
        alert_pattern = r'Price Alert[:\s]*([A-Z]{1,5})\s+(?:reached|hit|crossed)\s*\$?([\d,.]+)'
        alert_matches = re.findall(alert_pattern, body, re.IGNORECASE)
        
        for symbol, price in alert_matches:
            if len(symbol) >= 2:
                try:
                    current_price = float(price.replace(',', ''))
                    signals.append(EmailSignal(
                        symbol=symbol,
                        signal_type='alert',
                        source='webull',
                        subject=subject,
                        current_price=current_price,
                        confidence=0.7,
                        reasoning=f"Price alert triggered at ${current_price}",
                        email_date=date
                    ))
                except ValueError:
                    pass
        
        return signals
    
    def _parse_generic(self, email: Dict) -> List[EmailSignal]:
        """Generic parser for unknown newsletter formats."""
        signals = []
        body = email.get('body', '')
        subject = email.get('subject', '')
        date = email.get('date', '')
        
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_symbols = set(re.findall(stock_pattern, body))
        
        common_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'HAVE', 'WILL',
                       'YOUR', 'MORE', 'WHEN', 'WHAT', 'WERE', 'BEEN', 'THEIR', 'WOULD',
                       'COULD', 'ABOUT', 'INTO', 'ONLY', 'OTHER', 'NEW', 'SOME', 'TIME',
                       'CEO', 'CFO', 'IPO', 'SEC', 'FDA', 'NYSE', 'USA', 'USD', 'EUR'}
        
        for symbol in potential_symbols:
            if symbol not in common_words and len(symbol) >= 2:
                signal_type = self._detect_sentiment(body, symbol)
                if signal_type != 'neutral':
                    signals.append(EmailSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        source='generic',
                        subject=subject,
                        confidence=0.4,
                        reasoning="Detected in newsletter content",
                        email_date=date
                    ))
        
        return signals
    
    def _detect_sentiment(self, text: str, symbol: str) -> str:
        """Detect sentiment around a symbol mention in text."""
        symbol_pattern = rf'\b{symbol}\b'
        matches = list(re.finditer(symbol_pattern, text, re.IGNORECASE))
        
        if not matches:
            return 'neutral'
        
        bullish_score = 0
        bearish_score = 0
        
        for match in matches:
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text[start:end].lower()
            
            for keyword in self.BULLISH_KEYWORDS:
                if keyword in context:
                    bullish_score += 1
            
            for keyword in self.BEARISH_KEYWORDS:
                if keyword in context:
                    bearish_score += 1
        
        if bullish_score > bearish_score:
            return 'bullish'
        elif bearish_score > bullish_score:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get a summary of all parsed signals."""
        if not self.signals:
            return {'total': 0, 'by_source': {}, 'by_signal': {}, 'symbols': []}
        
        by_source = {}
        by_signal = {}
        symbols = set()
        
        for signal in self.signals:
            by_source[signal.source] = by_source.get(signal.source, 0) + 1
            by_signal[signal.signal_type] = by_signal.get(signal.signal_type, 0) + 1
            symbols.add(signal.symbol)
        
        return {
            'total': len(self.signals),
            'by_source': by_source,
            'by_signal': by_signal,
            'symbols': list(symbols),
            'bullish_count': by_signal.get('bullish', 0) + by_signal.get('buy', 0) + by_signal.get('strong_buy', 0),
            'bearish_count': by_signal.get('bearish', 0) + by_signal.get('sell', 0) + by_signal.get('strong_sell', 0)
        }
    
    def get_top_signals(self, min_confidence: float = 0.5) -> List[EmailSignal]:
        """Get signals with confidence above threshold."""
        return [s for s in self.signals if s.confidence >= min_confidence]
    
    def get_signals_by_symbol(self, symbol: str) -> List[EmailSignal]:
        """Get all signals for a specific symbol."""
        return [s for s in self.signals if s.symbol.upper() == symbol.upper()]
