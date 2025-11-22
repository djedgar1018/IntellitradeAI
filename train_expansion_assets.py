"""
Train AI models for 20 new assets (10 crypto + 10 stocks)
"""

from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion
import time

# 10 new cryptocurrencies (expanding from top 10 to top 20)
NEW_CRYPTOS = ["AVAX", "SHIB", "TON", "DOT", "LINK", "MATIC", "BCH", "LTC", "UNI", "XLM"]

# 10 new stocks (expanding from 8 to 18)
NEW_STOCKS = ["WMT", "JNJ", "V", "BAC", "DIS", "NFLX", "INTC", "AMD", "CRM", "ORCL"]

print("="*70)
print("🚀 Training AI Models for 20 New Assets")
print("="*70)
print(f"📊 New Cryptocurrencies: {len(NEW_CRYPTOS)}")
print(f"📈 New Stocks: {len(NEW_STOCKS)}")
print("="*70)

trainer = RobustModelTrainer()
ing = DataIngestion()

success_count = 0
failed = []

# Train cryptocurrency models
print("\n🪙 TRAINING CRYPTOCURRENCY MODELS")
print("-"*70)
for i, symbol in enumerate(NEW_CRYPTOS, 1):
    print(f"\n[{i}/{len(NEW_CRYPTOS)}] Training {symbol}...")
    try:
        # Fetch data
        data = ing.fetch_crypto_data([symbol], period='2y', interval='1d')
        
        if not data or symbol not in data:
            print(f"❌ {symbol}: No data available")
            failed.append(f"{symbol} (crypto)")
            continue
        
        df = data[symbol]
        
        if len(df) < 100:
            print(f"❌ {symbol}: Insufficient data ({len(df)} rows)")
            failed.append(f"{symbol} (crypto)")
            continue
        
        # Train model
        result = trainer.run_comprehensive_training(
            df, 
            symbol, 
            algorithms=['random_forest']
        )
        
        if 'random_forest' in result and 'accuracy' in result['random_forest']:
            acc = result['random_forest']['accuracy']
            print(f"✅ {symbol}: Model trained (Accuracy: {acc:.1%})")
            success_count += 1
        else:
            print(f"❌ {symbol}: Training failed")
            failed.append(f"{symbol} (crypto)")
            
    except Exception as e:
        print(f"❌ {symbol}: Error - {str(e)}")
        failed.append(f"{symbol} (crypto)")
    
    time.sleep(1)  # Brief pause between trainings

# Train stock models
print("\n\n📈 TRAINING STOCK MODELS")
print("-"*70)
for i, symbol in enumerate(NEW_STOCKS, 1):
    print(f"\n[{i}/{len(NEW_STOCKS)}] Training {symbol}...")
    try:
        # Fetch data
        data = ing.fetch_stock_data([symbol], period='2y', interval='1d')
        
        if not data or symbol not in data:
            print(f"❌ {symbol}: No data available")
            failed.append(f"{symbol} (stock)")
            continue
        
        df = data[symbol]
        
        if len(df) < 100:
            print(f"❌ {symbol}: Insufficient data ({len(df)} rows)")
            failed.append(f"{symbol} (stock)")
            continue
        
        # Train model
        result = trainer.run_comprehensive_training(
            df, 
            symbol, 
            algorithms=['random_forest']
        )
        
        if 'random_forest' in result and 'accuracy' in result['random_forest']:
            acc = result['random_forest']['accuracy']
            print(f"✅ {symbol}: Model trained (Accuracy: {acc:.1%})")
            success_count += 1
        else:
            print(f"❌ {symbol}: Training failed")
            failed.append(f"{symbol} (stock)")
            
    except Exception as e:
        print(f"❌ {symbol}: Error - {str(e)}")
        failed.append(f"{symbol} (stock)")
    
    time.sleep(1)  # Brief pause between trainings

# Final summary
print("\n\n" + "="*70)
print("📊 TRAINING COMPLETE")
print("="*70)
print(f"✅ Successfully trained: {success_count}/{len(NEW_CRYPTOS) + len(NEW_STOCKS)}")
print(f"❌ Failed: {len(failed)}")

if failed:
    print("\nFailed assets:")
    for asset in failed:
        print(f"  - {asset}")

print("\n🎉 Expansion complete! Dashboard now supports 38 total assets.")
print("="*70)
