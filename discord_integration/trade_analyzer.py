"""Analyze trade history from Discord to establish bias and methodology."""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np

from discord_integration.trade_parser import ParsedTrade, TradeMessageParser


@dataclass
class TradingPattern:
    """Represents a detected trading pattern or bias."""
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    symbols: List[str]
    win_rate: float
    avg_profit_pct: float
    time_of_day: Optional[str] = None
    day_of_week: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TraderProfile:
    """Profile built from analyzing trade history."""
    total_trades: int
    win_rate: float
    avg_holding_period: float
    preferred_asset_types: Dict[str, float]
    preferred_symbols: List[str]
    preferred_actions: Dict[str, float]
    risk_tolerance: str
    trading_style: str
    time_preferences: Dict[str, float]
    patterns: List[TradingPattern]
    strengths: List[str]
    weaknesses: List[str]
    methodology: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['patterns'] = [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.patterns]
        return d


class TradeHistoryAnalyzer:
    """Analyze trade history to establish bias, methodology, and replication strategy."""
    
    def __init__(self):
        self.trades: List[ParsedTrade] = []
        self.profile: Optional[TraderProfile] = None
        
    def add_trades(self, trades: List[ParsedTrade]):
        """Add parsed trades for analysis."""
        self.trades.extend(trades)
        
    def clear_trades(self):
        """Clear all stored trades."""
        self.trades = []
        self.profile = None
    
    def analyze(self) -> TraderProfile:
        """Perform comprehensive analysis of trade history."""
        if not self.trades:
            return self._empty_profile()
        
        wins = [t for t in self.trades if t.outcome == 'win']
        losses = [t for t in self.trades if t.outcome == 'loss']
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        asset_prefs = self._analyze_asset_preferences()
        symbol_prefs = self._analyze_symbol_preferences()
        action_prefs = self._analyze_action_preferences()
        time_prefs = self._analyze_time_preferences()
        patterns = self._detect_patterns()
        methodology = self._extract_methodology()
        
        risk_tolerance = self._assess_risk_tolerance()
        trading_style = self._determine_trading_style()
        strengths = self._identify_strengths()
        weaknesses = self._identify_weaknesses()
        
        self.profile = TraderProfile(
            total_trades=len(self.trades),
            win_rate=win_rate,
            avg_holding_period=self._calculate_avg_holding_period(),
            preferred_asset_types=asset_prefs,
            preferred_symbols=symbol_prefs[:10],
            preferred_actions=action_prefs,
            risk_tolerance=risk_tolerance,
            trading_style=trading_style,
            time_preferences=time_prefs,
            patterns=patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            methodology=methodology
        )
        
        return self.profile
    
    def _empty_profile(self) -> TraderProfile:
        """Return empty profile when no trades available."""
        return TraderProfile(
            total_trades=0,
            win_rate=0,
            avg_holding_period=0,
            preferred_asset_types={},
            preferred_symbols=[],
            preferred_actions={},
            risk_tolerance='unknown',
            trading_style='unknown',
            time_preferences={},
            patterns=[],
            strengths=[],
            weaknesses=[],
            methodology={}
        )
    
    def _analyze_asset_preferences(self) -> Dict[str, float]:
        """Analyze preferred asset types."""
        asset_counts = defaultdict(int)
        for trade in self.trades:
            asset_counts[trade.asset_type] += 1
        
        total = len(self.trades)
        return {k: round(v / total * 100, 1) for k, v in sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)}
    
    def _analyze_symbol_preferences(self) -> List[str]:
        """Analyze most traded symbols."""
        symbol_counts = defaultdict(int)
        symbol_wins = defaultdict(int)
        
        for trade in self.trades:
            symbol_counts[trade.symbol] += 1
            if trade.outcome == 'win':
                symbol_wins[trade.symbol] += 1
        
        symbol_scores = {}
        for symbol, count in symbol_counts.items():
            win_rate = symbol_wins[symbol] / count if count > 0 else 0
            symbol_scores[symbol] = count * (1 + win_rate)
        
        return [s for s, _ in sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)]
    
    def _analyze_action_preferences(self) -> Dict[str, float]:
        """Analyze buy/sell/call/put preferences."""
        action_counts = defaultdict(int)
        for trade in self.trades:
            action_counts[trade.action] += 1
        
        total = len(self.trades)
        return {k: round(v / total * 100, 1) for k, v in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)}
    
    def _analyze_time_preferences(self) -> Dict[str, float]:
        """Analyze preferred trading times."""
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for trade in self.trades:
            hour_counts[trade.timestamp.hour] += 1
            day_counts[trade.timestamp.strftime('%A')] += 1
        
        total = len(self.trades)
        
        peak_hours = [h for h, c in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        peak_days = [d for d, c in sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        return {
            'peak_hours': peak_hours,
            'peak_days': peak_days,
            'hour_distribution': dict(sorted(hour_counts.items())),
            'day_distribution': dict(day_counts)
        }
    
    def _detect_patterns(self) -> List[TradingPattern]:
        """Detect recurring trading patterns."""
        patterns = []
        
        bullish_bias = self._detect_directional_bias()
        if bullish_bias:
            patterns.append(bullish_bias)
        
        sector_patterns = self._detect_sector_patterns()
        patterns.extend(sector_patterns)
        
        momentum_pattern = self._detect_momentum_pattern()
        if momentum_pattern:
            patterns.append(momentum_pattern)
        
        contrarian_pattern = self._detect_contrarian_pattern()
        if contrarian_pattern:
            patterns.append(contrarian_pattern)
        
        return patterns
    
    def _detect_directional_bias(self) -> Optional[TradingPattern]:
        """Detect if trader has bullish or bearish bias."""
        long_trades = [t for t in self.trades if t.action in ['buy', 'call']]
        short_trades = [t for t in self.trades if t.action in ['sell', 'put']]
        
        if not self.trades:
            return None
        
        long_pct = len(long_trades) / len(self.trades) * 100
        
        if long_pct >= 70:
            wins = [t for t in long_trades if t.outcome == 'win']
            win_rate = len(wins) / len(long_trades) * 100 if long_trades else 0
            
            return TradingPattern(
                pattern_type='directional_bias',
                description='Strong bullish bias - predominantly long/call positions',
                confidence=long_pct,
                frequency=len(long_trades),
                symbols=[t.symbol for t in long_trades[:10]],
                win_rate=win_rate,
                avg_profit_pct=self._calc_avg_profit(long_trades)
            )
        elif long_pct <= 30:
            wins = [t for t in short_trades if t.outcome == 'win']
            win_rate = len(wins) / len(short_trades) * 100 if short_trades else 0
            
            return TradingPattern(
                pattern_type='directional_bias',
                description='Strong bearish bias - predominantly short/put positions',
                confidence=100 - long_pct,
                frequency=len(short_trades),
                symbols=[t.symbol for t in short_trades[:10]],
                win_rate=win_rate,
                avg_profit_pct=self._calc_avg_profit(short_trades)
            )
        
        return None
    
    def _detect_sector_patterns(self) -> List[TradingPattern]:
        """Detect sector concentration patterns."""
        patterns = []
        
        tech_symbols = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'TSM', 'MU'}
        crypto_majors = {'BTC', 'ETH', 'XRP', 'SOL', 'ADA'}
        meme_coins = {'DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF', 'FLOKI'}
        
        sector_trades = {
            'tech_stocks': [t for t in self.trades if t.symbol in tech_symbols],
            'crypto_majors': [t for t in self.trades if t.symbol in crypto_majors],
            'meme_coins': [t for t in self.trades if t.symbol in meme_coins]
        }
        
        for sector, trades in sector_trades.items():
            if len(trades) >= 5:
                wins = [t for t in trades if t.outcome == 'win']
                win_rate = len(wins) / len(trades) * 100 if trades else 0
                
                if len(trades) / len(self.trades) >= 0.2:
                    patterns.append(TradingPattern(
                        pattern_type='sector_focus',
                        description=f'Concentrated trading in {sector.replace("_", " ")}',
                        confidence=len(trades) / len(self.trades) * 100,
                        frequency=len(trades),
                        symbols=list(set(t.symbol for t in trades))[:10],
                        win_rate=win_rate,
                        avg_profit_pct=self._calc_avg_profit(trades)
                    ))
        
        return patterns
    
    def _detect_momentum_pattern(self) -> Optional[TradingPattern]:
        """Detect if trader follows momentum."""
        momentum_indicators = ['momentum', 'breakout', 'trend', 'running', 'ripping', 'mooning']
        
        momentum_trades = [t for t in self.trades if any(ind in (t.reasoning or '').lower() for ind in momentum_indicators)]
        
        if len(momentum_trades) >= 5:
            wins = [t for t in momentum_trades if t.outcome == 'win']
            win_rate = len(wins) / len(momentum_trades) * 100
            
            return TradingPattern(
                pattern_type='momentum_following',
                description='Momentum-based trading strategy detected',
                confidence=len(momentum_trades) / len(self.trades) * 100,
                frequency=len(momentum_trades),
                symbols=list(set(t.symbol for t in momentum_trades))[:10],
                win_rate=win_rate,
                avg_profit_pct=self._calc_avg_profit(momentum_trades)
            )
        
        return None
    
    def _detect_contrarian_pattern(self) -> Optional[TradingPattern]:
        """Detect if trader is contrarian."""
        contrarian_indicators = ['dip', 'oversold', 'reversal', 'bottom', 'fear', 'capitulation']
        
        contrarian_trades = [t for t in self.trades if any(ind in (t.reasoning or '').lower() for ind in contrarian_indicators)]
        
        if len(contrarian_trades) >= 5:
            wins = [t for t in contrarian_trades if t.outcome == 'win']
            win_rate = len(wins) / len(contrarian_trades) * 100
            
            return TradingPattern(
                pattern_type='contrarian',
                description='Contrarian/dip-buying strategy detected',
                confidence=len(contrarian_trades) / len(self.trades) * 100,
                frequency=len(contrarian_trades),
                symbols=list(set(t.symbol for t in contrarian_trades))[:10],
                win_rate=win_rate,
                avg_profit_pct=self._calc_avg_profit(contrarian_trades)
            )
        
        return None
    
    def _calc_avg_profit(self, trades: List[ParsedTrade]) -> float:
        """Calculate average profit percentage for a set of trades."""
        profits = [t.profit_loss_pct for t in trades if t.profit_loss_pct is not None]
        return round(np.mean(profits), 2) if profits else 0
    
    def _calculate_avg_holding_period(self) -> float:
        """Estimate average holding period in hours."""
        return 24.0
    
    def _assess_risk_tolerance(self) -> str:
        """Assess trader's risk tolerance level."""
        option_pct = sum(1 for t in self.trades if t.asset_type == 'option') / len(self.trades) * 100 if self.trades else 0
        meme_symbols = {'DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF', 'FLOKI', 'MEME'}
        meme_pct = sum(1 for t in self.trades if t.symbol in meme_symbols) / len(self.trades) * 100 if self.trades else 0
        
        if option_pct > 50 or meme_pct > 30:
            return 'aggressive'
        elif option_pct > 20 or meme_pct > 15:
            return 'moderate-aggressive'
        elif option_pct > 5 or meme_pct > 5:
            return 'moderate'
        else:
            return 'conservative'
    
    def _determine_trading_style(self) -> str:
        """Determine overall trading style."""
        action_counts = defaultdict(int)
        for trade in self.trades:
            action_counts[trade.action] += 1
        
        if not self.trades:
            return 'unknown'
        
        long_pct = (action_counts.get('buy', 0) + action_counts.get('call', 0)) / len(self.trades) * 100
        option_pct = (action_counts.get('call', 0) + action_counts.get('put', 0)) / len(self.trades) * 100
        
        if option_pct > 50:
            if long_pct > 70:
                return 'aggressive_calls'
            elif long_pct < 30:
                return 'aggressive_puts'
            else:
                return 'options_swing'
        elif long_pct > 70:
            return 'bullish_long'
        elif long_pct < 30:
            return 'bearish_short'
        else:
            return 'balanced_swing'
    
    def _identify_strengths(self) -> List[str]:
        """Identify trader's strengths based on performance."""
        strengths = []
        
        if not self.trades:
            return strengths
        
        wins = [t for t in self.trades if t.outcome == 'win']
        win_rate = len(wins) / len(self.trades) * 100
        
        if win_rate >= 70:
            strengths.append('High overall win rate')
        elif win_rate >= 55:
            strengths.append('Above average win rate')
        
        symbol_wins = defaultdict(list)
        for trade in self.trades:
            if trade.outcome == 'win':
                symbol_wins[trade.symbol].append(trade)
        
        for symbol, wins_list in symbol_wins.items():
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            if len(symbol_trades) >= 5:
                symbol_win_rate = len(wins_list) / len(symbol_trades) * 100
                if symbol_win_rate >= 75:
                    strengths.append(f'Strong performance in {symbol}')
        
        action_wins = defaultdict(list)
        for trade in self.trades:
            if trade.outcome == 'win':
                action_wins[trade.action].append(trade)
        
        for action, wins_list in action_wins.items():
            action_trades = [t for t in self.trades if t.action == action]
            if len(action_trades) >= 5:
                action_win_rate = len(wins_list) / len(action_trades) * 100
                if action_win_rate >= 70:
                    strengths.append(f'Skilled at {action} trades')
        
        return strengths[:5]
    
    def _identify_weaknesses(self) -> List[str]:
        """Identify areas for improvement."""
        weaknesses = []
        
        if not self.trades:
            return weaknesses
        
        wins = [t for t in self.trades if t.outcome == 'win']
        win_rate = len(wins) / len(self.trades) * 100
        
        if win_rate < 40:
            weaknesses.append('Low overall win rate')
        
        symbol_losses = defaultdict(list)
        for trade in self.trades:
            if trade.outcome == 'loss':
                symbol_losses[trade.symbol].append(trade)
        
        for symbol, losses_list in symbol_losses.items():
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            if len(symbol_trades) >= 5:
                symbol_loss_rate = len(losses_list) / len(symbol_trades) * 100
                if symbol_loss_rate >= 60:
                    weaknesses.append(f'Poor performance in {symbol}')
        
        return weaknesses[:5]
    
    def _extract_methodology(self) -> Dict[str, Any]:
        """Extract trading methodology from patterns and trades."""
        methodology = {
            'entry_criteria': [],
            'exit_criteria': [],
            'position_sizing': 'unknown',
            'risk_management': [],
            'preferred_setups': []
        }
        
        if not self.trades:
            return methodology
        
        reasonings = [t.reasoning for t in self.trades if t.reasoning]
        
        entry_keywords = {
            'rsi': 'RSI-based entries',
            'breakout': 'Breakout trading',
            'support': 'Support level bounces',
            'dip': 'Buying dips',
            'momentum': 'Momentum following',
            'volume': 'Volume confirmation',
            'macd': 'MACD crossovers',
            'trend': 'Trend following'
        }
        
        for keyword, description in entry_keywords.items():
            if any(keyword in (r or '').lower() for r in reasonings):
                methodology['entry_criteria'].append(description)
        
        exit_keywords = {
            'target': 'Fixed profit targets',
            'stop': 'Stop-loss discipline',
            'resistance': 'Resistance exits',
            'overbought': 'Overbought exits',
            'trailing': 'Trailing stops'
        }
        
        for keyword, description in exit_keywords.items():
            if any(keyword in (r or '').lower() for r in reasonings):
                methodology['exit_criteria'].append(description)
        
        win_rate = len([t for t in self.trades if t.outcome == 'win']) / len(self.trades) * 100
        if win_rate >= 60:
            methodology['risk_management'].append('Positive expectancy')
        
        methodology['preferred_setups'] = self._identify_setups()
        
        return methodology
    
    def _identify_setups(self) -> List[str]:
        """Identify preferred trade setups."""
        setups = []
        
        if not self.trades:
            return setups
        
        option_pct = sum(1 for t in self.trades if t.asset_type == 'option') / len(self.trades) * 100
        if option_pct > 30:
            call_trades = [t for t in self.trades if t.action == 'call']
            put_trades = [t for t in self.trades if t.action == 'put']
            
            if len(call_trades) > len(put_trades) * 2:
                setups.append('Bullish call options')
            elif len(put_trades) > len(call_trades) * 2:
                setups.append('Bearish put options')
            else:
                setups.append('Balanced options strategies')
        
        crypto_pct = sum(1 for t in self.trades if t.asset_type == 'crypto') / len(self.trades) * 100
        if crypto_pct > 30:
            setups.append('Cryptocurrency swing trades')
        
        return setups
    
    def generate_replication_strategy(self) -> Dict[str, Any]:
        """Generate a strategy to replicate the analyzed trading methodology."""
        if not self.profile:
            self.analyze()
        
        if not self.profile or self.profile.total_trades == 0:
            return {'error': 'No trade history to analyze'}
        
        strategy = {
            'name': f'{self.profile.trading_style}_replication',
            'description': f'AI replication of {self.profile.trading_style} methodology',
            'parameters': {
                'directional_bias': 'long' if self.profile.trading_style in ['bullish_long', 'aggressive_calls'] else 'short' if self.profile.trading_style in ['bearish_short', 'aggressive_puts'] else 'neutral',
                'preferred_symbols': self.profile.preferred_symbols[:10],
                'asset_allocation': self.profile.preferred_asset_types,
                'risk_level': self.profile.risk_tolerance,
                'win_rate_target': self.profile.win_rate
            },
            'entry_rules': self.profile.methodology.get('entry_criteria', []),
            'exit_rules': self.profile.methodology.get('exit_criteria', []),
            'risk_management': {
                'max_position_size': 10 if self.profile.risk_tolerance == 'aggressive' else 5 if self.profile.risk_tolerance == 'moderate' else 3,
                'stop_loss': 15 if self.profile.risk_tolerance == 'aggressive' else 10 if self.profile.risk_tolerance == 'moderate' else 5,
                'take_profit': 30 if self.profile.risk_tolerance == 'aggressive' else 20 if self.profile.risk_tolerance == 'moderate' else 10
            },
            'time_filters': {
                'preferred_hours': self.profile.time_preferences.get('peak_hours', []),
                'preferred_days': self.profile.time_preferences.get('peak_days', [])
            },
            'pattern_weights': {}
        }
        
        for pattern in self.profile.patterns:
            strategy['pattern_weights'][pattern.pattern_type] = {
                'weight': pattern.confidence / 100,
                'win_rate': pattern.win_rate,
                'symbols': pattern.symbols
            }
        
        return strategy
    
    def save_analysis(self, filepath: str):
        """Save analysis results to file."""
        if not self.profile:
            self.analyze()
        
        data = {
            'analyzed_at': datetime.now().isoformat(),
            'total_trades': len(self.trades),
            'profile': self.profile.to_dict() if self.profile else None,
            'replication_strategy': self.generate_replication_strategy(),
            'trades': [t.to_dict() for t in self.trades[:100]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_analysis(self, filepath: str) -> Optional[Dict]:
        """Load previous analysis from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
