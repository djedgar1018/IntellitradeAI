"""
Simple and Direct ML Model Training for Top 10 Cryptocurrencies
Trains Random Forest models for all top 10 coins including XRP
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.top_coins_manager import TopCoinsManager
from data.enhanced_crypto_fetcher import EnhancedCryptoFetcher

def calculate_technical_indicators(df):
    """Calculate technical indicators for features"""
    df = df.copy()
    
    # Price changes
    df['return'] = df['close'].pct_change()
    df['high_low_pct'] = (df['high'] - df['low']) / df['low']
    
    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(4)
    
    # Volatility
    df['volatility'] = df['return'].rolling(window=20).std()
    
    return df

def prepare_features_and_target(df):
    """Prepare features (X) and target (y) for training"""
    # Calculate all indicators
    df = calculate_technical_indicators(df)
    
    # Create target: 1 if price goes up tomorrow, 0 otherwise
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN rows
    df = df.dropna()
    
    # Feature columns
    feature_cols = [
        'return', 'high_low_pct',
        'ma_5', 'ma_10', 'ma_20',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_middle', 'bb_lower',
        'volume_change', 'volume_ma',
        'momentum', 'volatility'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, df

def train_model(symbol, df):
    """Train a Random Forest model for a given cryptocurrency"""
    print(f"\n   Training Random Forest for {symbol}...")
    
    try:
        # Prepare features
        X, y, processed_df = prepare_features_and_target(df)
        
        if len(X) < 50:
            print(f"   ⚠️  Not enough data ({len(X)} samples), need at least 50")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"   → Training set: {len(X_train)} samples")
        print(f"   → Test set: {len(X_test)} samples")
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Save model
        model_dir = 'models/cache'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{symbol}_random_forest.joblib')
        
        joblib.dump({
            'model': model,
            'feature_columns': list(X.columns),
            'trained_date': datetime.now().isoformat(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }, model_path)
        
        print(f"   ✅ Model trained and saved!")
        print(f"      • Accuracy:  {accuracy:.2%}")
        print(f"      • Precision: {precision:.2%}")
        print(f"      • Recall:    {recall:.2%}")
        print(f"      • F1 Score:  {f1:.2%}")
        
        return {
            'symbol': symbol,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def main():
    """Main training function"""
    
    print("\n" + "="*80)
    print("🤖 AI MODEL TRAINING - TOP 10 CRYPTOCURRENCIES (INCLUDING XRP)")
    print("="*80)
    
    # Step 1: Get top 10 coins
    print("\n📊 STEP 1: Fetching Top 10 Cryptocurrencies...")
    coins_manager = TopCoinsManager()
    top_coins = coins_manager.fetch_top_coins(limit=10)
    
    symbols = [coin['symbol'] for coin in top_coins]
    print(f"✅ Found {len(symbols)} coins: {', '.join(symbols)}")
    
    # Highlight XRP
    if 'XRP' in symbols:
        print(f"   🎯 XRP is #{symbols.index('XRP') + 1} in the list")
    
    # Step 2: Fetch historical data
    print("\n📈 STEP 2: Fetching Historical Price Data (6 months)...")
    fetcher = EnhancedCryptoFetcher()
    data_dict = fetcher.fetch_top_n_coins_data(n=10, period='6mo')
    
    print(f"✅ Successfully fetched data for {len(data_dict)} coins")
    
    # Step 3: Train models
    print("\n🧠 STEP 3: Training AI Models...")
    print(f"   • Algorithm: Random Forest")
    print(f"   • Features: 15 technical indicators")
    print(f"   • Target: Predict if price goes UP or DOWN tomorrow")
    print("-" * 80)
    
    results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        if symbol not in data_dict or data_dict[symbol].empty:
            print(f"   ❌ No data available")
            continue
        
        df = data_dict[symbol]
        print(f"   → Data points: {len(df)}")
        
        result = train_model(symbol, df)
        if result:
            results.append(result)
    
    # Step 4: Display Summary
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    
    if results:
        print(f"\n✅ Successfully trained {len(results)}/{len(symbols)} models\n")
        
        # Create summary table
        print(f"{'Symbol':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Train/Test'}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['symbol']:<10} "
                  f"{r['accuracy']:<12.2%} "
                  f"{r['precision']:<12.2%} "
                  f"{r['recall']:<12.2%} "
                  f"{r['f1_score']:<12.2%} "
                  f"{r['train_samples']}/{r['test_samples']}")
        
        # Average performance
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        print("-" * 80)
        print(f"{'AVERAGE':<10} "
              f"{avg_accuracy:<12.2%} "
              f"{avg_precision:<12.2%} "
              f"{avg_recall:<12.2%} "
              f"{avg_f1:<12.2%}")
        
        # XRP specific message
        xrp_result = next((r for r in results if r['symbol'] == 'XRP'), None)
        if xrp_result:
            print("\n" + "="*80)
            print("🎯 XRP MODEL READY!")
            print("="*80)
            print(f"   ✅ XRP model trained with {xrp_result['accuracy']:.1%} accuracy")
            print(f"   ✅ You can now get AI predictions for XRP!")
            print(f"   ✅ Model saved to: models/cache/XRP_random_forest.joblib")
    else:
        print("\n❌ No models were successfully trained")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    if results:
        print("\n💾 Models saved to: models/cache/")
        print("\n🚀 You can now use these models for:")
        print("   • BUY/SELL/HOLD signal generation")
        print("   • Price direction predictions")
        print("   • Portfolio optimization")
        print("   • Backtesting strategies")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
