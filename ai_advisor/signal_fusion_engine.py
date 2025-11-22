"""
Signal Fusion Engine
Intelligently combines ML predictions and chart pattern signals
Resolves conflicts and provides unified trading recommendations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ai_advisor.price_level_analyzer import PriceLevelAnalyzer


@dataclass
class SignalPayload:
    """Unified signal structure for all AI systems"""
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    source: str  # 'ML_MODEL' or 'PATTERN_RECOGNITION'
    reasoning: str  # Explanation for the signal
    technical_data: Dict  # Additional technical details


class SignalFusionEngine:
    """
    Combines multiple AI signal sources into a unified recommendation
    Handles conflicts and provides transparent reasoning
    """
    
    def __init__(self):
        # Weighting factors for different signal sources
        self.ml_base_weight = 0.6  # ML model gets 60% base weight
        self.pattern_base_weight = 0.4  # Pattern recognition gets 40% base weight
        
        # Conflict thresholds
        self.high_confidence_threshold = 0.65
        self.conflict_gap_threshold = 0.15
        
        # Price level analyzer for HOLD signals
        self.price_analyzer = PriceLevelAnalyzer()
    
    def fuse_signals(self, ml_prediction: Dict, pattern_signals: List[Dict], 
                    symbol: str, historical_data=None) -> Dict:
        """
        Combine ML prediction and pattern signals into unified recommendation
        
        Args:
            ml_prediction: Output from MLPredictor
            pattern_signals: List of patterns from ChartPatternRecognizer
            symbol: Cryptocurrency symbol
        
        Returns:
            Unified signal with conflict resolution
        """
        # Convert inputs to SignalPayload format
        ml_signal = self._convert_ml_to_payload(ml_prediction)
        pattern_signal = self._convert_patterns_to_payload(pattern_signals)
        
        # Check for conflicts
        has_conflict = self._detect_conflict(ml_signal, pattern_signal)
        
        if has_conflict:
            # Resolve conflict
            unified_signal = self._resolve_conflict(ml_signal, pattern_signal, ml_prediction, symbol, historical_data)
        else:
            # No conflict - combine strengths
            unified_signal = self._combine_aligned_signals(ml_signal, pattern_signal, ml_prediction, historical_data)
        
        # Add both perspectives for transparency
        unified_signal['ml_insight'] = {
            'signal': ml_signal.signal,
            'confidence': ml_signal.confidence,
            'reasoning': ml_signal.reasoning
        }
        
        unified_signal['pattern_insight'] = {
            'signal': pattern_signal.signal,
            'confidence': pattern_signal.confidence,
            'reasoning': pattern_signal.reasoning
        }
        
        unified_signal['has_conflict'] = has_conflict
        
        return unified_signal
    
    def _convert_ml_to_payload(self, ml_pred: Dict) -> SignalPayload:
        """Convert ML prediction to SignalPayload format"""
        return SignalPayload(
            signal=ml_pred.get('signal', 'HOLD'),
            confidence=ml_pred.get('confidence', 0.5),
            source='ML_MODEL',
            reasoning=ml_pred.get('recommendation', {}).get('action_explanation', 'ML-based prediction'),
            technical_data=ml_pred.get('technical_indicators', {})
        )
    
    def _convert_patterns_to_payload(self, patterns: List[Dict]) -> SignalPayload:
        """Convert pattern signals to SignalPayload format"""
        if not patterns:
            return SignalPayload(
                signal='HOLD',
                confidence=0.0,
                source='PATTERN_RECOGNITION',
                reasoning='No chart patterns detected',
                technical_data={}
            )
        
        # Use the strongest pattern signal
        strongest = max(patterns, key=lambda p: p.get('confidence', 0))
        
        return SignalPayload(
            signal=strongest.get('signal', 'HOLD'),
            confidence=strongest.get('confidence', 0.5),
            source='PATTERN_RECOGNITION',
            reasoning=f"{strongest.get('pattern_type', 'Pattern')}: {strongest.get('description', 'Detected pattern')}",
            technical_data=strongest
        )
    
    def _detect_conflict(self, ml_signal: SignalPayload, pattern_signal: SignalPayload) -> bool:
        """Detect if ML and pattern signals conflict"""
        # HOLD doesn't conflict with anything
        if ml_signal.signal == 'HOLD' or pattern_signal.signal == 'HOLD':
            return False
        
        # BUY vs SELL is a conflict
        if ml_signal.signal != pattern_signal.signal:
            return True
        
        return False
    
    def _resolve_conflict(self, ml_signal: SignalPayload, pattern_signal: SignalPayload,
                         ml_prediction: Dict, symbol: str, historical_data=None) -> Dict:
        """
        Resolve conflicting signals with intelligent logic
        
        Rules:
        1. If both have high confidence (>65%) but gap <15% → HOLD (too risky)
        2. If gap ≥15% → Choose higher confidence signal
        3. Weight ML model by its actual accuracy for this symbol
        """
        ml_conf = ml_signal.confidence
        pattern_conf = pattern_signal.confidence
        
        # Get ML model's actual accuracy for weighting
        ml_metrics = ml_prediction.get('model_metrics', {})
        ml_accuracy = ml_metrics.get('accuracy', 0.5)  # Default to 50% if unknown
        
        # Adjust weights based on actual model performance
        ml_weight = self.ml_base_weight * (ml_accuracy / 0.5)  # Scale by accuracy
        pattern_weight = self.pattern_base_weight
        
        # Normalize weights
        total_weight = ml_weight + pattern_weight
        ml_weight /= total_weight
        pattern_weight /= total_weight
        
        # Calculate weighted scores
        ml_score = ml_conf * ml_weight
        pattern_score = pattern_conf * pattern_weight
        
        # Check confidence gap
        confidence_gap = abs(ml_conf - pattern_conf)
        
        # Rule 1: Both high confidence but close gap → HOLD (avoid risky conflict)
        if (ml_conf >= self.high_confidence_threshold and 
            pattern_conf >= self.high_confidence_threshold and
            confidence_gap < self.conflict_gap_threshold):
            
            hold_signal = {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': max(ml_conf, pattern_conf),
                'recommendation': {
                    'decision': 'HOLD',
                    'confidence_level': 'High',
                    'risk_level': 'High',
                    'action_explanation': (
                        f"⚠️ CONFLICTING SIGNALS DETECTED: ML Model says {ml_signal.signal} "
                        f"({ml_conf:.1%} confidence) but Chart Pattern shows {pattern_signal.signal} "
                        f"({pattern_conf:.1%} confidence). Both signals are strong but contradictory. "
                        f"Recommendation: HOLD until signals align. This protects you from risky trades "
                        f"when AI systems disagree."
                    )
                },
                'conflict_reason': f'High-confidence disagreement: {ml_signal.signal} vs {pattern_signal.signal}',
                'current_price': ml_prediction.get('current_price', 0),
                'price_change_24h': ml_prediction.get('price_change_24h', 0),
                'technical_indicators': ml_prediction.get('technical_indicators', {})
            }
            
            # Add price levels for HOLD signals
            if historical_data is not None:
                price_levels = self.price_analyzer.analyze_key_levels(
                    historical_data,
                    hold_signal['current_price'],
                    hold_signal['technical_indicators']
                )
                hold_signal['price_levels'] = price_levels
            
            return hold_signal
        
        # Rule 2: Choose higher weighted score
        if ml_score > pattern_score:
            chosen_signal = ml_signal.signal
            chosen_conf = ml_conf
            reason = (
                f"ML Model prediction ({ml_signal.signal} at {ml_conf:.1%}) has higher weighted "
                f"confidence than Pattern Recognition ({pattern_signal.signal} at {pattern_conf:.1%}). "
                f"However, note that Chart Pattern detected: {pattern_signal.reasoning}. "
                f"Exercise caution due to conflicting signals."
            )
        else:
            chosen_signal = pattern_signal.signal
            chosen_conf = pattern_conf
            reason = (
                f"Chart Pattern ({pattern_signal.signal} at {pattern_conf:.1%}) has higher confidence "
                f"than ML Model ({ml_signal.signal} at {ml_conf:.1%}). "
                f"However, ML analysis suggests: {ml_signal.reasoning}. "
                f"Exercise caution due to conflicting signals."
            )
        
        # Determine confidence and risk levels
        if chosen_conf >= 0.75:
            confidence_level = 'High'
        elif chosen_conf >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'symbol': symbol,
            'signal': chosen_signal,
            'confidence': chosen_conf,
            'recommendation': {
                'decision': chosen_signal,
                'confidence_level': confidence_level,
                'risk_level': 'High',  # Always high risk when conflicting
                'action_explanation': f"⚠️ {reason}"
            },
            'conflict_reason': f'Resolved: Chose {chosen_signal} (ML: {ml_signal.signal}, Pattern: {pattern_signal.signal})',
            'current_price': ml_prediction.get('current_price', 0),
            'price_change_24h': ml_prediction.get('price_change_24h', 0),
            'technical_indicators': ml_prediction.get('technical_indicators', {})
        }
    
    def _combine_aligned_signals(self, ml_signal: SignalPayload, 
                                pattern_signal: SignalPayload, ml_prediction: Dict, historical_data=None) -> Dict:
        """
        Combine signals when they agree (both BUY, both SELL, or one is HOLD)
        """
        # If signals agree, boost confidence
        if ml_signal.signal == pattern_signal.signal and ml_signal.signal != 'HOLD':
            # Both agree on BUY or SELL - boost confidence
            combined_confidence = min(0.95, (ml_signal.confidence + pattern_signal.confidence) / 2 * 1.1)
            signal = ml_signal.signal
            
            explanation = (
                f"✅ ALIGNED SIGNALS: Both ML Model and Chart Pattern agree on {signal}. "
                f"ML confidence: {ml_signal.confidence:.1%}, Pattern confidence: {pattern_signal.confidence:.1%}. "
                f"Combined confidence: {combined_confidence:.1%}. Strong agreement across both AI systems."
            )
            
        elif ml_signal.signal == 'HOLD' or pattern_signal.signal == 'HOLD':
            # One says HOLD - use the other signal but don't boost
            if ml_signal.signal != 'HOLD':
                signal = ml_signal.signal
                combined_confidence = ml_signal.confidence * 0.9  # Slight penalty for no pattern confirmation
                explanation = f"ML Model suggests {signal} ({ml_signal.confidence:.1%}). Chart patterns are neutral."
            else:
                signal = pattern_signal.signal
                combined_confidence = pattern_signal.confidence * 0.9
                explanation = f"Chart Pattern suggests {signal} ({pattern_signal.confidence:.1%}). ML model is neutral."
        else:
            # Both HOLD
            signal = 'HOLD'
            combined_confidence = max(ml_signal.confidence, pattern_signal.confidence)
            explanation = "Both ML Model and Chart Pattern suggest waiting. No clear signals detected. Watch the key levels below for actionable opportunities."
        
        # Determine levels
        if combined_confidence >= 0.75:
            confidence_level = 'High'
        elif combined_confidence >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        volatility = ml_prediction.get('technical_indicators', {}).get('volatility', 0.03)
        if volatility > 0.05:
            risk_level = 'High'
        elif volatility > 0.03:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        result = {
            'symbol': ml_prediction.get('symbol', 'UNKNOWN'),
            'signal': signal,
            'confidence': combined_confidence,
            'recommendation': {
                'decision': signal,
                'confidence_level': confidence_level,
                'risk_level': risk_level,
                'action_explanation': explanation
            },
            'conflict_reason': None,
            'current_price': ml_prediction.get('current_price', 0),
            'price_change_24h': ml_prediction.get('price_change_24h', 0),
            'technical_indicators': ml_prediction.get('technical_indicators', {})
        }
        
        # Add price levels for HOLD signals
        if signal == 'HOLD' and historical_data is not None:
            price_levels = self.price_analyzer.analyze_key_levels(
                historical_data,
                result['current_price'],
                result['technical_indicators']
            )
            result['price_levels'] = price_levels
        
        return result
