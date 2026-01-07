"""Main Discord service for IntelliTradeAI - orchestrates message fetching, parsing, and analysis."""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

from discord_integration.client import DiscordClient
from discord_integration.trade_parser import TradeMessageParser, ParsedTrade
from discord_integration.trade_analyzer import TradeHistoryAnalyzer, TraderProfile


class DiscordTradingService:
    """Main service for analyzing Discord trading conversations."""
    
    def __init__(self, cache_dir: str = "data/discord_cache"):
        self.client = DiscordClient()
        self.parser = TradeMessageParser()
        self.analyzer = TradeHistoryAnalyzer()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._configured_channel_id: Optional[str] = None
        self._configured_guild_id: Optional[str] = None
        
    def configure_channel(self, guild_id: str, channel_id: str):
        """Configure which channel/thread to analyze."""
        self._configured_guild_id = guild_id
        self._configured_channel_id = channel_id
        
        config = {
            'guild_id': guild_id,
            'channel_id': channel_id,
            'configured_at': datetime.now().isoformat()
        }
        
        with open(self.cache_dir / 'channel_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_channel_config(self) -> Optional[Dict]:
        """Load saved channel configuration."""
        config_path = self.cache_dir / 'channel_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._configured_guild_id = config.get('guild_id')
                self._configured_channel_id = config.get('channel_id')
                return config
        return None
    
    def test_connection(self) -> Dict:
        """Test Discord connection and return status."""
        return self.client.test_connection()
    
    def get_available_guilds(self) -> List[Dict]:
        """Get list of available Discord servers."""
        return self.client.get_guilds()
    
    def get_guild_channels(self, guild_id: str) -> List[Dict]:
        """Get channels in a guild."""
        return self.client.get_channels(guild_id)
    
    def get_trading_channels(self, guild_id: str) -> List[Dict]:
        """Get channels that likely contain trading discussions."""
        return self.client.find_trading_channels(guild_id)
    
    def fetch_and_analyze_history(self, channel_id: str = None, days: int = 365) -> Dict:
        """Fetch message history and perform full analysis."""
        channel_id = channel_id or self._configured_channel_id
        if not channel_id:
            return {'error': 'No channel configured. Call configure_channel first.'}
        
        print(f"Fetching {days} days of message history from channel {channel_id}...")
        
        messages = self.client.get_message_history(channel_id, days=days)
        print(f"Fetched {len(messages)} messages")
        
        self._cache_messages(channel_id, messages)
        
        trades = self.parser.parse_messages(messages)
        print(f"Parsed {len(trades)} trades from messages")
        
        self.analyzer.clear_trades()
        self.analyzer.add_trades(trades)
        
        profile = self.analyzer.analyze()
        
        strategy = self.analyzer.generate_replication_strategy()
        
        analysis_path = self.cache_dir / f'analysis_{channel_id}_{datetime.now().strftime("%Y%m%d")}.json'
        self.analyzer.save_analysis(str(analysis_path))
        
        return {
            'messages_fetched': len(messages),
            'trades_parsed': len(trades),
            'profile': profile.to_dict() if profile else None,
            'replication_strategy': strategy,
            'analysis_saved_to': str(analysis_path)
        }
    
    def _cache_messages(self, channel_id: str, messages: List[Dict]):
        """Cache messages to disk for offline analysis."""
        cache_file = self.cache_dir / f'messages_{channel_id}.json'
        with open(cache_file, 'w') as f:
            json.dump({
                'channel_id': channel_id,
                'fetched_at': datetime.now().isoformat(),
                'message_count': len(messages),
                'messages': messages
            }, f)
    
    def load_cached_messages(self, channel_id: str) -> Optional[List[Dict]]:
        """Load cached messages from disk."""
        cache_file = self.cache_dir / f'messages_{channel_id}.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('messages', [])
        return None
    
    def get_daily_analysis(self, channel_id: str = None) -> Dict:
        """Perform daily analysis of recent trades."""
        channel_id = channel_id or self._configured_channel_id
        if not channel_id:
            return {'error': 'No channel configured'}
        
        messages = self.client.get_message_history(channel_id, days=1)
        trades = self.parser.parse_messages(messages)
        
        trade_summary = self.parser.get_trade_summary(trades)
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'channel_id': channel_id,
            'messages_analyzed': len(messages),
            'trades_found': len(trades),
            'summary': trade_summary,
            'trades': [t.to_dict() for t in trades]
        }
    
    def get_signal_bias(self, symbol: str) -> Dict:
        """Get trading bias for a symbol based on historical Discord trades."""
        if not self.analyzer.trades:
            cached = self.load_cached_messages(self._configured_channel_id or '')
            if cached:
                trades = self.parser.parse_messages(cached)
                self.analyzer.add_trades(trades)
        
        symbol_trades = [t for t in self.analyzer.trades if t.symbol.upper() == symbol.upper()]
        
        if not symbol_trades:
            return {
                'symbol': symbol,
                'bias': 'neutral',
                'confidence': 0,
                'trade_count': 0
            }
        
        wins = [t for t in symbol_trades if t.outcome == 'win']
        losses = [t for t in symbol_trades if t.outcome == 'loss']
        
        long_trades = [t for t in symbol_trades if t.action in ['buy', 'call']]
        short_trades = [t for t in symbol_trades if t.action in ['sell', 'put']]
        
        long_wins = [t for t in long_trades if t.outcome == 'win']
        short_wins = [t for t in short_trades if t.outcome == 'win']
        
        long_win_rate = len(long_wins) / len(long_trades) * 100 if long_trades else 0
        short_win_rate = len(short_wins) / len(short_trades) * 100 if short_trades else 0
        
        if long_win_rate > short_win_rate + 10:
            bias = 'bullish'
            confidence = long_win_rate
        elif short_win_rate > long_win_rate + 10:
            bias = 'bearish'
            confidence = short_win_rate
        else:
            bias = 'neutral'
            confidence = max(long_win_rate, short_win_rate)
        
        return {
            'symbol': symbol,
            'bias': bias,
            'confidence': round(confidence, 1),
            'trade_count': len(symbol_trades),
            'win_rate': round(len(wins) / len(symbol_trades) * 100, 1) if symbol_trades else 0,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': round(long_win_rate, 1),
            'short_win_rate': round(short_win_rate, 1)
        }
    
    def get_current_methodology(self) -> Dict:
        """Get the current learned trading methodology."""
        if not self.analyzer.profile:
            latest_analysis = self._load_latest_analysis()
            if latest_analysis:
                return {
                    'methodology': latest_analysis.get('profile', {}).get('methodology', {}),
                    'replication_strategy': latest_analysis.get('replication_strategy', {}),
                    'last_updated': latest_analysis.get('analyzed_at')
                }
            return {'error': 'No analysis available. Run fetch_and_analyze_history first.'}
        
        return {
            'methodology': self.analyzer.profile.methodology,
            'replication_strategy': self.analyzer.generate_replication_strategy(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _load_latest_analysis(self) -> Optional[Dict]:
        """Load the most recent analysis file."""
        analysis_files = list(self.cache_dir.glob('analysis_*.json'))
        if not analysis_files:
            return None
        
        latest = max(analysis_files, key=lambda f: f.stat().st_mtime)
        with open(latest, 'r') as f:
            return json.load(f)
    
    def integrate_with_signals(self, signals: List[Dict]) -> List[Dict]:
        """Enhance trading signals with Discord-learned bias."""
        enhanced_signals = []
        
        for signal in signals:
            symbol = signal.get('symbol', '')
            bias = self.get_signal_bias(symbol)
            
            enhanced_signal = signal.copy()
            enhanced_signal['discord_bias'] = bias.get('bias', 'neutral')
            enhanced_signal['discord_confidence'] = bias.get('confidence', 0)
            enhanced_signal['discord_win_rate'] = bias.get('win_rate', 0)
            
            if bias.get('bias') == 'bullish' and signal.get('signal') == 'BUY':
                enhanced_signal['confidence'] = min(100, signal.get('confidence', 50) + bias.get('confidence', 0) * 0.2)
            elif bias.get('bias') == 'bearish' and signal.get('signal') == 'SELL':
                enhanced_signal['confidence'] = min(100, signal.get('confidence', 50) + bias.get('confidence', 0) * 0.2)
            
            enhanced_signals.append(enhanced_signal)
        
        return enhanced_signals


def create_discord_service() -> DiscordTradingService:
    """Factory function to create Discord service."""
    return DiscordTradingService()
