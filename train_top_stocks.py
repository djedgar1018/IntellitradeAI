"""
Train AI models for the TOP most popular stocks
"""

import time
from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion

# Top most popular stocks users will want
TOP_STOCKS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft  
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
    'META',   # Meta
    'TSLA',   # Tesla
    'JPM',    # JPMorgan
    'V',      # Visa
    'MA',     # Mastercard
    'WMT',    # Walmart
    'JNJ',    # Johnson & Johnson
    'XOM',    # Exxon
    'AMD',    # AMD
    'NFLX',   # Netflix
]

def train_one_stock(symbol, trainer, ing):
    """Train model for one stock"""
    try:
        print(f"\n{'='*60}")
        print(f"Training {symbol}...")
        print(f"{'='*60}")
        
        # Fetch data
        data = ing.fetch_stock_data([symbol], period='1y', interval='1d')
        if not data or symbol not in data:
            print(f"❌ No data for {symbol}")
            return False
        
        stock_data = data[symbol]
        print(f"✅ Fetched {len(stock_data)} rows")
        
        # Train
        result = trainer.run_comprehensive_training(
            data=stock_data,
            symbol=symbol,
            algorithms=['random_forest'],
            optimize_hyperparams=True
        )
        
        if result and result.get('models', {}).get('random_forest', {}).get('status') == 'success':
            acc = result['models']['random_forest']['accuracy']
            print(f"✅ {symbol} trained! Accuracy: {acc:.1%}")
            return True
        else:
            print(f"❌ {symbol} training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    trainer = RobustModelTrainer()
    ing = DataIngestion()
    
    print("="*60)
    print(f"Training {len(TOP_STOCKS)} most popular stocks")
    print("="*60)
    
    success_count = 0
    for i, symbol in enumerate(TOP_STOCKS, 1):
        print(f"\n[{i}/{len(TOP_STOCKS)}]")
        if train_one_stock(symbol, trainer, ing):
            success_count += 1
        time.sleep(0.5)  # Small delay
    
    print("\n" + "="*60)
    print(f"✅ Successfully trained: {success_count}/{len(TOP_STOCKS)}")
    print("="*60)
