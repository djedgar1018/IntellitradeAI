"""
Train AI Models for Top 10 Cryptocurrencies
Trains Random Forest and XGBoost models for all top 10 coins
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.top_coins_manager import TopCoinsManager
from data.enhanced_crypto_fetcher import EnhancedCryptoFetcher
from models.model_trainer import RobustModelTrainer

def train_all_top10_models():
    """Train models for all top 10 cryptocurrencies"""
    
    print("\n" + "="*70)
    print("🤖 AI MODEL TRAINING - TOP 10 CRYPTOCURRENCIES")
    print("="*70)
    
    # Step 1: Get top 10 coins
    print("\n📊 Step 1: Fetching Top 10 Cryptocurrencies...")
    coins_manager = TopCoinsManager()
    top_coins = coins_manager.fetch_top_coins(limit=10)
    
    symbols = [coin['symbol'] for coin in top_coins]
    print(f"✅ Found {len(symbols)} coins: {', '.join(symbols)}")
    
    # Step 2: Fetch historical data
    print("\n📈 Step 2: Fetching Historical Price Data (6 months)...")
    fetcher = EnhancedCryptoFetcher()
    data_dict = fetcher.fetch_top_n_coins_data(n=10, period='6mo')
    
    print(f"✅ Successfully fetched data for {len(data_dict)} coins")
    for symbol, df in data_dict.items():
        print(f"   • {symbol}: {len(df)} data points")
    
    # Step 3: Train models for each coin
    print("\n🧠 Step 3: Training AI Models...")
    print("   Algorithms: Random Forest & XGBoost")
    print("   Features: 70+ technical indicators")
    print("-" * 70)
    
    trainer = RobustModelTrainer()
    results = {
        'successful': [],
        'failed': [],
        'metrics': {}
    }
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Training models for {symbol}...")
        
        if symbol not in data_dict or data_dict[symbol].empty:
            print(f"❌ No data available for {symbol}, skipping...")
            results['failed'].append(symbol)
            continue
        
        try:
            df = data_dict[symbol]
            
            # Train Random Forest
            print(f"   → Training Random Forest for {symbol}...")
            rf_result = trainer.train_random_forest(
                data=df,
                symbol=symbol,
                model_type='classifier'
            )
            
            if rf_result and rf_result.get('model'):
                print(f"   ✅ Random Forest trained successfully!")
                if 'metrics' in rf_result:
                    print(f"      Accuracy: {rf_result['metrics'].get('accuracy', 0):.2%}")
                    print(f"      F1 Score: {rf_result['metrics'].get('f1_score', 0):.2%}")
                
                # Train XGBoost
                print(f"   → Training XGBoost for {symbol}...")
                xgb_result = trainer.train_xgboost(
                    data=df,
                    symbol=symbol,
                    model_type='classifier'
                )
                
                if xgb_result and xgb_result.get('model'):
                    print(f"   ✅ XGBoost trained successfully!")
                    if 'metrics' in xgb_result:
                        print(f"      Accuracy: {xgb_result['metrics'].get('accuracy', 0):.2%}")
                        print(f"      F1 Score: {xgb_result['metrics'].get('f1_score', 0):.2%}")
                    
                    results['successful'].append(symbol)
                    results['metrics'][symbol] = {
                        'random_forest': rf_result.get('metrics', {}),
                        'xgboost': xgb_result.get('metrics', {})
                    }
                else:
                    print(f"   ⚠️  XGBoost training failed")
                    results['failed'].append(symbol)
            else:
                print(f"   ⚠️  Random Forest training failed")
                results['failed'].append(symbol)
                
        except Exception as e:
            print(f"   ❌ Error training {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            results['failed'].append(symbol)
            continue
    
    # Step 4: Display Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n✅ Successfully Trained: {len(results['successful'])}/{len(symbols)} coins")
    if results['successful']:
        print(f"   {', '.join(results['successful'])}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])} coins")
        print(f"   {', '.join(results['failed'])}")
    
    # Step 5: Performance Table
    if results['metrics']:
        print("\n📈 MODEL PERFORMANCE METRICS")
        print("-" * 70)
        print(f"{'Symbol':<10} {'Algorithm':<15} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12}")
        print("-" * 70)
        
        for symbol, metrics in results['metrics'].items():
            for algo, scores in metrics.items():
                algo_name = 'Random Forest' if algo == 'random_forest' else 'XGBoost'
                print(f"{symbol:<10} {algo_name:<15} "
                      f"{scores['accuracy']:<12.2%} "
                      f"{scores['f1_score']:<12.2%} "
                      f"{scores['precision']:<12.2%}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    return results


if __name__ == '__main__':
    try:
        results = train_all_top10_models()
        
        print("\n💾 Models saved to: models/cache/")
        print("\n🚀 You can now use these models for:")
        print("   • Real-time price predictions")
        print("   • BUY/SELL/HOLD signal generation")
        print("   • Portfolio optimization")
        print("   • Backtesting strategies")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
