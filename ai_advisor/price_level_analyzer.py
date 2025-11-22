"""
Price Level Analyzer
Calculates key support and resistance levels for actionable trading decisions
Provides recommendations for what to do at each price level
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class PriceLevelAnalyzer:
    """
    Analyzes price data to identify key support and resistance levels
    Provides actionable recommendations for each level
    """
    
    def __init__(self):
        self.lookback_period = 50  # Days to look back for swing points
        
    def analyze_key_levels(self, df: pd.DataFrame, current_price: float, 
                          technical_indicators: Dict) -> Dict:
        """
        Calculate 3 key levels (support and resistance) with actionable recommendations
        
        Args:
            df: Historical OHLCV data
            current_price: Current asset price
            technical_indicators: RSI, MACD, volatility, etc.
        
        Returns:
            Dictionary with key levels and recommendations
        """
        # Calculate support and resistance levels
        support_levels = self._calculate_support_levels(df, current_price)
        resistance_levels = self._calculate_resistance_levels(df, current_price)
        
        # Get volatility for risk assessment
        volatility = technical_indicators.get('volatility', 0.03)
        rsi = technical_indicators.get('rsi', 50)
        
        # Combine and prioritize levels
        all_levels = []
        
        # Add resistance levels (upside targets) - these are SELL opportunities
        for level in resistance_levels[:3]:
            distance_pct = ((level - current_price) / current_price) * 100
            all_levels.append({
                'price': level,
                'type': 'RESISTANCE',
                'distance_pct': distance_pct,
                'action': 'SELL',
                'confidence': self._calculate_level_confidence(level, df, 'resistance'),
                'reasoning': self._generate_level_reasoning(
                    'SELL', level, current_price, distance_pct, volatility, rsi
                )
            })
        
        # Add support levels (downside targets) - these are BUY opportunities
        for level in support_levels[:3]:
            distance_pct = ((level - current_price) / current_price) * 100
            all_levels.append({
                'price': level,
                'type': 'SUPPORT',
                'distance_pct': distance_pct,
                'action': 'BUY',
                'confidence': self._calculate_level_confidence(level, df, 'support'),
                'reasoning': self._generate_level_reasoning(
                    'BUY', level, current_price, distance_pct, volatility, rsi
                )
            })
        
        # Sort by absolute distance to get nearest levels
        all_levels.sort(key=lambda x: abs(x['distance_pct']))
        
        # Take top 3 levels (closest to current price)
        key_levels = all_levels[:3]
        
        # Fallback: If we have fewer than 3 levels, synthesize additional ones using volatility bands
        if len(key_levels) < 3:
            volatility = technical_indicators.get('volatility', 0.03)
            fallback_levels = self._synthesize_fallback_levels(
                current_price, volatility, 3 - len(key_levels)
            )
            key_levels.extend(fallback_levels)
        
        # Sort by distance (negative = below, positive = above)
        key_levels.sort(key=lambda x: x['distance_pct'])
        
        return {
            'key_levels': key_levels,
            'current_price': current_price,
            'nearest_support': support_levels[0] if support_levels else current_price * 0.95,
            'nearest_resistance': resistance_levels[0] if resistance_levels else current_price * 1.05,
            'volatility': volatility
        }
    
    def _calculate_support_levels(self, df: pd.DataFrame, current_price: float) -> List[float]:
        """Calculate support levels from swing lows and technical indicators"""
        supports = []
        
        # Get recent data
        recent_df = df.tail(self.lookback_period)
        
        # Method 1: Swing lows (local minima)
        swing_lows = self._find_swing_lows(recent_df)
        supports.extend([low for low in swing_lows if low < current_price])
        
        # Method 2: Previous resistance becomes support
        swing_highs = self._find_swing_highs(recent_df)
        supports.extend([high for high in swing_highs if high < current_price])
        
        # Method 3: Round number support (psychological levels)
        round_supports = self._find_round_number_levels(current_price, direction='down')
        supports.extend(round_supports)
        
        # Method 4: Moving average support
        if 'close' in recent_df.columns and len(recent_df) >= 10:
            ma_20 = recent_df['close'].rolling(window=20, min_periods=10).mean().iloc[-1]
            if not np.isnan(ma_20) and ma_20 < current_price and ma_20 > 0:
                supports.append(ma_20)
            
            if len(recent_df) >= 25:
                ma_50 = recent_df['close'].rolling(window=50, min_periods=25).mean().iloc[-1]
                if not np.isnan(ma_50) and ma_50 < current_price and ma_50 > 0:
                    supports.append(ma_50)
        
        # Remove duplicates and sort by proximity to current price
        supports = list(set([round(s, 2) for s in supports if s > 0]))
        supports.sort(reverse=True)  # Closest support first
        
        return supports[:5]  # Return top 5 support levels
    
    def _calculate_resistance_levels(self, df: pd.DataFrame, current_price: float) -> List[float]:
        """Calculate resistance levels from swing highs and technical indicators"""
        resistances = []
        
        # Get recent data
        recent_df = df.tail(self.lookback_period)
        
        # Method 1: Swing highs (local maxima)
        swing_highs = self._find_swing_highs(recent_df)
        resistances.extend([high for high in swing_highs if high > current_price])
        
        # Method 2: Previous support becomes resistance
        swing_lows = self._find_swing_lows(recent_df)
        resistances.extend([low for low in swing_lows if low > current_price])
        
        # Method 3: Round number resistance (psychological levels)
        round_resistances = self._find_round_number_levels(current_price, direction='up')
        resistances.extend(round_resistances)
        
        # Method 4: Moving average resistance
        if 'close' in recent_df.columns and len(recent_df) >= 10:
            ma_20 = recent_df['close'].rolling(window=20, min_periods=10).mean().iloc[-1]
            if not np.isnan(ma_20) and ma_20 > current_price and ma_20 > 0:
                resistances.append(ma_20)
            
            if len(recent_df) >= 25:
                ma_50 = recent_df['close'].rolling(window=50, min_periods=25).mean().iloc[-1]
                if not np.isnan(ma_50) and ma_50 > current_price and ma_50 > 0:
                    resistances.append(ma_50)
        
        # Remove duplicates and sort by proximity to current price
        resistances = list(set([round(r, 2) for r in resistances if r > 0]))
        resistances.sort()  # Closest resistance first
        
        return resistances[:5]  # Return top 5 resistance levels
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing low points (local minima)"""
        if 'low' not in df.columns or len(df) < window + 2:
            return []
        
        lows = df['low'].values
        swing_lows = []
        
        for i in range(window, len(lows) - window):
            # Check if this point is lower than surrounding points
            if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
                swing_lows.append(float(lows[i]))
        
        return swing_lows
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing high points (local maxima)"""
        if 'high' not in df.columns or len(df) < window + 2:
            return []
        
        highs = df['high'].values
        swing_highs = []
        
        for i in range(window, len(highs) - window):
            # Check if this point is higher than surrounding points
            if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
                swing_highs.append(float(highs[i]))
        
        return swing_highs
    
    def _find_round_number_levels(self, current_price: float, direction: str = 'both') -> List[float]:
        """Find psychological round number levels"""
        levels = []
        
        # Determine appropriate intervals based on price magnitude
        if current_price < 1:
            intervals = [0.01, 0.05, 0.1, 0.5]
        elif current_price < 10:
            intervals = [0.1, 0.5, 1, 5]
        elif current_price < 100:
            intervals = [1, 5, 10, 50]
        elif current_price < 1000:
            intervals = [10, 50, 100, 500]
        else:
            intervals = [100, 500, 1000, 5000]
        
        for interval in intervals:
            # Find nearest round number below
            if direction in ['down', 'both']:
                level_down = (current_price // interval) * interval
                if level_down < current_price and level_down > 0:
                    levels.append(level_down)
            
            # Find nearest round number above
            if direction in ['up', 'both']:
                level_up = ((current_price // interval) + 1) * interval
                if level_up > current_price:
                    levels.append(level_up)
        
        return sorted(list(set(levels)))[:3]  # Return top 3
    
    def _calculate_level_confidence(self, level: float, df: pd.DataFrame, 
                                    level_type: str) -> float:
        """
        Calculate confidence in a support/resistance level
        Based on how many times price has bounced off this level
        """
        if len(df) < 20:
            return 0.5
        
        # Count touches near this level (within 2%)
        tolerance = 0.02
        touches = 0
        
        if level_type == 'support':
            for low in df['low'].tail(50):
                if abs(low - level) / level < tolerance:
                    touches += 1
        else:  # resistance
            for high in df['high'].tail(50):
                if abs(high - level) / level < tolerance:
                    touches += 1
        
        # More touches = higher confidence (cap at 0.9)
        confidence = min(0.5 + (touches * 0.1), 0.9)
        return confidence
    
    def _synthesize_fallback_levels(self, current_price: float, volatility: float, 
                                    num_needed: int) -> List[Dict]:
        """
        Synthesize price levels using volatility bands when insufficient historical data
        Ensures we always have at least 3 levels to show users
        """
        fallback_levels = []
        
        # Use volatility to create bands
        # Typically 1 standard deviation move
        volatility_move = current_price * volatility
        
        # Create support level (below current price)
        if len(fallback_levels) < num_needed:
            support_price = current_price - (volatility_move * 1.5)
            distance_pct = ((support_price - current_price) / current_price) * 100
            
            fallback_levels.append({
                'price': round(support_price, 2),
                'type': 'SUPPORT',
                'distance_pct': distance_pct,
                'action': 'BUY',
                'confidence': 0.6,  # Lower confidence for synthesized levels
                'reasoning': f"Volatility-based support at ${support_price:.2f} ({distance_pct:.1f}% below). Consider buying if price drops to this level."
            })
        
        # Create resistance level (above current price)
        if len(fallback_levels) < num_needed:
            resistance_price = current_price + (volatility_move * 1.5)
            distance_pct = ((resistance_price - current_price) / current_price) * 100
            
            fallback_levels.append({
                'price': round(resistance_price, 2),
                'type': 'RESISTANCE',
                'distance_pct': distance_pct,
                'action': 'SELL',
                'confidence': 0.6,
                'reasoning': f"Volatility-based resistance at ${resistance_price:.2f} (+{distance_pct:.1f}%). Consider selling if price rises to this level."
            })
        
        # If still need more, add wider bands
        if len(fallback_levels) < num_needed:
            wider_support = current_price - (volatility_move * 2.5)
            distance_pct = ((wider_support - current_price) / current_price) * 100
            
            fallback_levels.append({
                'price': round(wider_support, 2),
                'type': 'SUPPORT',
                'distance_pct': distance_pct,
                'action': 'BUY',
                'confidence': 0.5,
                'reasoning': f"Extended support zone at ${wider_support:.2f} ({distance_pct:.1f}% below). Deep dip buy opportunity."
            })
        
        return fallback_levels[:num_needed]
    
    def _generate_level_reasoning(self, action: str, target_price: float, 
                                  current_price: float, distance_pct: float,
                                  volatility: float, rsi: float) -> str:
        """Generate human-readable reasoning for the recommended action"""
        
        if action == 'BUY':
            # Support level - buy opportunity
            if distance_pct >= 0:
                return f"Currently trading at support. Consider buying now with stop-loss below ${target_price:.2f}."
            else:
                abs_distance = abs(distance_pct)
                if abs_distance < 2:
                    return f"Strong support at ${target_price:.2f} ({abs_distance:.1f}% below). Good buy opportunity if it dips."
                elif abs_distance < 5:
                    return f"Key support at ${target_price:.2f} ({abs_distance:.1f}% below). Excellent buy zone if price reaches this level."
                else:
                    return f"Major support at ${target_price:.2f} ({abs_distance:.1f}% below). Wait for price to drop to this level before buying."
        
        else:  # SELL
            # Resistance level - sell opportunity
            if distance_pct <= 0:
                return f"Currently at resistance. Consider taking profits now before potential reversal."
            else:
                if distance_pct < 2:
                    return f"Approaching resistance at ${target_price:.2f} (+{distance_pct:.1f}%). Consider selling if it reaches this level."
                elif distance_pct < 5:
                    return f"Next resistance at ${target_price:.2f} (+{distance_pct:.1f}%). Good profit-taking opportunity if price rises there."
                else:
                    return f"Major resistance at ${target_price:.2f} (+{distance_pct:.1f}%). Strong sell target if price rallies to this level."
