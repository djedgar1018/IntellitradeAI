"""
Test HOLD signal with Price Level Analysis
Verify that key levels are calculated and actionable recommendations are provided
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_advisor.signal_fusion_engine import SignalFusionEngine

# Create realistic price data with clear support/resistance levels
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

# Simulate price with clear levels
base_price = 50
prices = []
for i in range(100):
    # Add oscillation around key levels (45, 50, 55)
    if i < 30:
        prices.append(base_price + np.random.randn() * 2)  # Around 50
    elif i < 60:
        prices.append(55 + np.random.randn() * 2)  # Rally to 55 (resistance)
    else:
        prices.append(base_price + np.random.randn() * 2)  # Back to 50

historical_data = pd.DataFrame({
    'open': prices,
    'high': [p + abs(np.random.randn()) for p in prices],
    'low': [p - abs(np.random.randn()) for p in prices],
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, 100)
}, index=dates)

current_price = prices[-1]

# Create a conflict scenario that results in HOLD
ml_prediction = {
    'symbol': 'TEST',
    'signal': 'BUY',
    'confidence': 0.68,  # High confidence BUY
    'current_price': current_price,
    'price_change_24h': -1.2,
    'recommendation': {
        'decision': 'BUY',
        'confidence_level': 'Medium',
        'risk_level': 'Medium',
        'action_explanation': 'ML model predicts upward movement.'
    },
    'technical_indicators': {
        'rsi': 45,
        'macd': 0.5,
        'ma_5': 49.5,
        'ma_10': 50.2,
        'ma_20': 51.0,
        'volatility': 0.035
    },
    'model_metrics': {
        'accuracy': 0.55
    }
}

# Pattern says SELL (conflict!)
pattern_signals = [{
    'pattern_type': 'Bearish Double Top',
    'signal': 'SELL',
    'confidence': 0.70,  # High confidence SELL
    'description': 'Double top at $55 resistance level'
}]

print("="*80)
print("🧪 HOLD SIGNAL WITH PRICE LEVELS TEST")
print("="*80)
print()
print("📊 Test Setup:")
print(f"  • Current Price: ${current_price:.2f}")
print(f"  • ML Signal: BUY (68% confidence)")
print(f"  • Pattern Signal: SELL (70% confidence)")
print(f"  • Expected Result: HOLD (conflict)")
print()
print("-"*80)

# Create fusion engine and fuse signals
fusion_engine = SignalFusionEngine()
unified_signal = fusion_engine.fuse_signals(
    ml_prediction=ml_prediction,
    pattern_signals=pattern_signals,
    symbol='TEST',
    historical_data=historical_data  # THIS IS KEY - provides data for price level analysis
)

print()
print("🎯 UNIFIED SIGNAL RESULT:")
print(f"  • Final Signal: {unified_signal['signal']}")
print(f"  • Confidence: {unified_signal['confidence']:.1%}")
print(f"  • Has Conflict: {unified_signal['has_conflict']}")
print()

if unified_signal['signal'] == 'HOLD':
    print("✅ HOLD signal generated as expected!")
    print()
    
    if 'price_levels' in unified_signal:
        print("="*80)
        print("🎯 KEY PRICE LEVELS - ACTIONABLE TRADING PLAN")
        print("="*80)
        
        price_levels_data = unified_signal['price_levels']
        key_levels = price_levels_data['key_levels']
        
        print()
        print(f"Current Price: ${price_levels_data['current_price']:.2f}")
        print(f"Nearest Support: ${price_levels_data['nearest_support']:.2f}")
        print(f"Nearest Resistance: ${price_levels_data['nearest_resistance']:.2f}")
        print()
        print("-"*80)
        print()
        
        for i, level in enumerate(key_levels, 1):
            action_icon = {'BUY': '📈 BUY', 'SELL': '📉 SELL'}.get(level['action'], '⏸️ WAIT')
            type_icon = {'SUPPORT': '🛡️', 'RESISTANCE': '🚧'}.get(level['type'], '📍')
            
            print(f"{type_icon} LEVEL {i}: ${level['price']:.2f} ({level['distance_pct']:+.1f}%)")
            print(f"   Action: {action_icon}")
            print(f"   Type: {level['type']}")
            print(f"   Confidence: {level['confidence']:.0%}")
            print(f"   ℹ️  {level['reasoning']}")
            print()
        
        print("="*80)
        print("✅ TEST PASSED: Price levels successfully calculated!")
        print("="*80)
    else:
        print("❌ FAIL: Price levels NOT found in unified signal!")
else:
    print(f"❌ FAIL: Expected HOLD but got {unified_signal['signal']}")

print()
print("📝 Full Recommendation:")
print(f"   {unified_signal['recommendation']['action_explanation']}")
print()
