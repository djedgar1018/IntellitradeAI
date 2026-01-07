"""Email trading service for analyzing newsletter signals."""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from email_integration.imap_client import IMAPClient
from email_integration.newsletter_parser import NewsletterParser, EmailSignal


class EmailTradingService:
    """Service for fetching and analyzing trading signals from emails."""
    
    def __init__(self):
        self.client = IMAPClient()
        self.parser = NewsletterParser()
        self.cache_dir = Path("email_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def test_connection(self) -> Dict:
        """Test the Gmail connection."""
        return self.client.test_connection()
    
    def fetch_and_analyze(self, days: int = 30) -> Dict[str, Any]:
        """Fetch newsletter emails and analyze for trading signals."""
        print(f"\nFetching newsletter emails from the past {days} days...")
        
        emails = self.client.get_newsletter_emails(days=days)
        print(f"Retrieved {len(emails)} emails")
        
        if not emails:
            return {
                'success': False,
                'error': 'No newsletter emails found',
                'emails_count': 0,
                'signals': []
            }
        
        self._cache_emails(emails)
        
        print("Parsing emails for trading signals...")
        signals = self.parser.parse_emails(emails)
        print(f"Extracted {len(signals)} trading signals")
        
        summary = self.parser.get_signal_summary()
        
        self._save_signals(signals)
        
        return {
            'success': True,
            'emails_count': len(emails),
            'signals_count': len(signals),
            'summary': summary,
            'signals': [self._signal_to_dict(s) for s in signals],
            'top_bullish': self._get_top_symbols('bullish'),
            'top_bearish': self._get_top_symbols('bearish')
        }
    
    def fetch_by_source(self, source: str, days: int = 30) -> Dict[str, Any]:
        """Fetch emails from a specific source."""
        source_map = {
            'tldr': 'tldr@tldrnewsletter.com',
            'barchart': 'barchart.com',
            'investing': 'investing.com',
            'webull': 'webull.com'
        }
        
        sender = source_map.get(source.lower())
        if not sender:
            return {'success': False, 'error': f'Unknown source: {source}'}
        
        print(f"\nFetching {source} emails...")
        emails = self.client.get_emails_by_sender(sender, days=days)
        print(f"Retrieved {len(emails)} emails from {source}")
        
        if not emails:
            return {
                'success': False,
                'error': f'No emails found from {source}',
                'emails_count': 0
            }
        
        signals = self.parser.parse_emails(emails)
        
        return {
            'success': True,
            'source': source,
            'emails_count': len(emails),
            'signals_count': len(signals),
            'signals': [self._signal_to_dict(s) for s in signals]
        }
    
    def get_symbol_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get all newsletter intelligence for a specific symbol."""
        signals = self.parser.get_signals_by_symbol(symbol)
        
        if not signals:
            return {
                'symbol': symbol,
                'has_signals': False,
                'message': f'No newsletter signals found for {symbol}'
            }
        
        bullish = sum(1 for s in signals if s.signal_type in ['bullish', 'buy', 'strong_buy'])
        bearish = sum(1 for s in signals if s.signal_type in ['bearish', 'sell', 'strong_sell'])
        
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        sources = list(set(s.source for s in signals))
        
        price_targets = [s.price_target for s in signals if s.price_target]
        avg_target = sum(price_targets) / len(price_targets) if price_targets else None
        
        if bullish > bearish:
            overall_bias = 'bullish'
        elif bearish > bullish:
            overall_bias = 'bearish'
        else:
            overall_bias = 'neutral'
        
        return {
            'symbol': symbol,
            'has_signals': True,
            'signal_count': len(signals),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'overall_bias': overall_bias,
            'avg_confidence': avg_confidence,
            'sources': sources,
            'avg_price_target': avg_target,
            'latest_signals': [self._signal_to_dict(s) for s in signals[-5:]]
        }
    
    def get_actionable_signals(self, min_confidence: float = 0.6) -> List[Dict]:
        """Get high-confidence actionable signals."""
        top_signals = self.parser.get_top_signals(min_confidence)
        
        symbol_signals = {}
        for signal in top_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        actionable = []
        for symbol, signals in symbol_signals.items():
            bullish = sum(1 for s in signals if s.signal_type in ['bullish', 'buy', 'strong_buy'])
            bearish = sum(1 for s in signals if s.signal_type in ['bearish', 'sell', 'strong_sell'])
            
            if bullish > 0 or bearish > 0:
                avg_conf = sum(s.confidence for s in signals) / len(signals)
                
                actionable.append({
                    'symbol': symbol,
                    'direction': 'long' if bullish > bearish else 'short',
                    'strength': abs(bullish - bearish),
                    'confidence': avg_conf,
                    'sources': list(set(s.source for s in signals)),
                    'signal_count': len(signals)
                })
        
        actionable.sort(key=lambda x: (x['strength'], x['confidence']), reverse=True)
        
        return actionable[:20]
    
    def _get_top_symbols(self, direction: str, limit: int = 10) -> List[Dict]:
        """Get top symbols by direction."""
        symbol_scores = {}
        
        for signal in self.parser.signals:
            if direction == 'bullish' and signal.signal_type in ['bullish', 'buy', 'strong_buy']:
                score = signal.confidence
            elif direction == 'bearish' and signal.signal_type in ['bearish', 'sell', 'strong_sell']:
                score = signal.confidence
            else:
                continue
            
            if signal.symbol not in symbol_scores:
                symbol_scores[signal.symbol] = {'score': 0, 'count': 0, 'sources': set()}
            
            symbol_scores[signal.symbol]['score'] += score
            symbol_scores[signal.symbol]['count'] += 1
            symbol_scores[signal.symbol]['sources'].add(signal.source)
        
        sorted_symbols = sorted(
            symbol_scores.items(),
            key=lambda x: (x[1]['score'], x[1]['count']),
            reverse=True
        )
        
        return [
            {
                'symbol': sym,
                'score': data['score'],
                'mentions': data['count'],
                'sources': list(data['sources'])
            }
            for sym, data in sorted_symbols[:limit]
        ]
    
    def _signal_to_dict(self, signal: EmailSignal) -> Dict:
        """Convert EmailSignal to dictionary."""
        return {
            'symbol': signal.symbol,
            'signal_type': signal.signal_type,
            'source': signal.source,
            'subject': signal.subject[:100] if signal.subject else '',
            'price_target': signal.price_target,
            'current_price': signal.current_price,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
            'asset_type': signal.asset_type,
            'email_date': signal.email_date
        }
    
    def _cache_emails(self, emails: List[Dict]):
        """Cache emails locally."""
        cache_file = self.cache_dir / f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cache_data = []
        for email in emails:
            cache_data.append({
                'id': email.get('id'),
                'from': email.get('from'),
                'subject': email.get('subject'),
                'date': email.get('date'),
                'snippet': email.get('snippet', '')[:200]
            })
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _save_signals(self, signals: List[EmailSignal]):
        """Save extracted signals."""
        signals_file = self.cache_dir / f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(signals_file, 'w') as f:
            json.dump([self._signal_to_dict(s) for s in signals], f, indent=2)


def run_email_analysis():
    """CLI function to run email analysis."""
    service = EmailTradingService()
    
    print("=" * 60)
    print("Email Newsletter Trading Signal Analyzer")
    print("=" * 60)
    
    conn = service.test_connection()
    print(f"\nGmail connection: {'OK' if conn.get('connected') else 'FAILED'}")
    if conn.get('connected'):
        print(f"Email: {conn.get('email')}")
    else:
        print(f"Error: {conn.get('error')}")
        return
    
    result = service.fetch_and_analyze(days=30)
    
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    if result.get('success'):
        print(f"Emails analyzed: {result['emails_count']}")
        print(f"Signals extracted: {result['signals_count']}")
        
        summary = result.get('summary', {})
        print(f"\nBy source: {summary.get('by_source', {})}")
        print(f"Bullish signals: {summary.get('bullish_count', 0)}")
        print(f"Bearish signals: {summary.get('bearish_count', 0)}")
        
        print("\nTop Bullish Symbols:")
        for item in result.get('top_bullish', [])[:5]:
            print(f"  {item['symbol']}: score={item['score']:.2f}, mentions={item['mentions']}")
        
        print("\nTop Bearish Symbols:")
        for item in result.get('top_bearish', [])[:5]:
            print(f"  {item['symbol']}: score={item['score']:.2f}, mentions={item['mentions']}")
        
        print("\nActionable Signals:")
        actionable = service.get_actionable_signals()
        for sig in actionable[:10]:
            print(f"  {sig['symbol']}: {sig['direction'].upper()} (confidence={sig['confidence']:.2f}, sources={sig['sources']})")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    run_email_analysis()
