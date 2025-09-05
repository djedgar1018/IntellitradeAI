"""
Simplified Chart Pattern Recognition System
For demonstrating trading signals and pattern detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class ChartPatternRecognizer:
    """
    AI-powered chart pattern recognition system
    """
    
    def __init__(self):
        # Pattern definitions with success rates
        self.pattern_library = {
            'Bullish Double Bottom': {
                'type': 'reversal',
                'signal': 'BUY',
                'success_rate': 0.78,
                'description': 'Two troughs at similar levels followed by rise'
            },
            'Bearish Double Top': {
                'type': 'reversal',
                'signal': 'SELL',
                'success_rate': 0.78,
                'description': 'Two peaks at similar levels followed by decline'
            },
            'Bullish Flag Pattern': {
                'type': 'continuation',
                'signal': 'BUY',
                'success_rate': 0.83,
                'description': 'Brief consolidation after strong upward move'
            },
            'Bearish Flag Pattern': {
                'type': 'continuation',
                'signal': 'SELL',
                'success_rate': 0.83,
                'description': 'Brief consolidation after strong downward move'
            },
            'Ascending Triangle': {
                'type': 'continuation',
                'signal': 'BUY',
                'success_rate': 0.72,
                'description': 'Horizontal resistance with rising support'
            }
        }
    
    def detect_patterns_from_data(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, any]]:
        """
        Detect chart patterns from OHLCV data
        """
        if len(df) < 10:
            return []
        
        patterns = []
        current_price = df['close'].iloc[-1]
        
        # Simple pattern detection based on price movements
        returns = df['close'].pct_change().fillna(0)
        
        # Detect trend patterns
        recent_trend = returns.tail(5).mean()
        volatility = returns.std()
        
        if recent_trend > 0.02:
            # Strong upward movement - potential bullish pattern
            pattern = self._create_pattern_signal(
                'Bullish Flag Pattern', 
                current_price, 
                symbol,
                confidence=min(0.9, 0.7 + recent_trend * 10)
            )
            patterns.append(pattern)
            
        elif recent_trend < -0.02:
            # Strong downward movement - potential bearish pattern
            pattern = self._create_pattern_signal(
                'Bearish Flag Pattern', 
                current_price, 
                symbol,
                confidence=min(0.9, 0.7 + abs(recent_trend) * 10)
            )
            patterns.append(pattern)
            
        else:
            # Sideways movement - potential continuation pattern
            pattern = self._create_pattern_signal(
                'Ascending Triangle', 
                current_price, 
                symbol,
                confidence=0.65
            )
            patterns.append(pattern)
        
        return patterns
    
    def _create_pattern_signal(self, pattern_name: str, current_price: float, 
                              symbol: str, confidence: float = 0.75) -> Dict[str, any]:
        """
        Create a pattern signal with trading levels
        """
        pattern_info = self.pattern_library.get(pattern_name, {})
        
        # Calculate trading levels based on pattern
        if pattern_info.get('signal') == 'BUY':
            target_price = current_price * 1.08  # 8% target
            stop_loss = current_price * 0.96     # 4% stop
        elif pattern_info.get('signal') == 'SELL':
            target_price = current_price * 0.92  # 8% target down
            stop_loss = current_price * 1.04     # 4% stop up
        else:
            target_price = current_price * 1.05  # 5% either direction
            stop_loss = current_price * 0.98     # 2% stop
        
        return {
            'symbol': symbol,
            'pattern_type': pattern_name,
            'signal': pattern_info.get('signal', 'HOLD'),
            'confidence': confidence,
            'success_rate': pattern_info.get('success_rate', 0.70),
            'signal_strength': confidence * pattern_info.get('success_rate', 0.70),
            'entry_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'risk_reward_ratio': abs(target_price - current_price) / abs(stop_loss - current_price),
            'description': pattern_info.get('description', 'Pattern detected'),
            'detected_at': datetime.utcnow().isoformat(),
            'timeframe': '1D',
            'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }