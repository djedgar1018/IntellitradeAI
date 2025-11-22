"""
Batch Training Script for Stock Models
Train AI models for a comprehensive list of popular stocks
"""

import sys
import time
from datetime import datetime
from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion

# Comprehensive list of popular stocks across sectors
STOCK_LIST = {
    'Technology': [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet (Google)
        'AMZN',   # Amazon
        'META',   # Meta (Facebook)
        'NVDA',   # NVIDIA
        'TSLA',   # Tesla
        'AMD',    # AMD
        'INTC',   # Intel
        'ORCL',   # Oracle
        'ADBE',   # Adobe
        'CRM',    # Salesforce
        'NFLX',   # Netflix
    ],
    'Finance': [
        'JPM',    # JPMorgan Chase
        'BAC',    # Bank of America
        'WFC',    # Wells Fargo
        'GS',     # Goldman Sachs
        'MS',     # Morgan Stanley
        'V',      # Visa
        'MA',     # Mastercard
    ],
    'Healthcare': [
        'JNJ',    # Johnson & Johnson
        'UNH',    # UnitedHealth
        'PFE',    # Pfizer
        'ABBV',   # AbbVie
        'MRK',    # Merck
        'LLY',    # Eli Lilly
    ],
    'Consumer': [
        'WMT',    # Walmart
        'HD',     # Home Depot
        'NKE',    # Nike
        'COST',   # Costco
        'MCD',    # McDonald's
        'SBUX',   # Starbucks
        'DIS',    # Disney
    ],
    'Energy': [
        'XOM',    # Exxon Mobil
        'CVX',    # Chevron
    ],
    'Industrial': [
        'BA',     # Boeing
        'CAT',    # Caterpillar
        'GE',     # General Electric
    ]
}

def get_all_stocks():
    """Flatten the stock list"""
    all_stocks = []
    for sector, stocks in STOCK_LIST.items():
        all_stocks.extend(stocks)
    return all_stocks

def train_stock_model(symbol, trainer, data_ingestion):
    """
    Train a model for a single stock
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        print(f"\n{'='*80}")
        print(f"📊 Training model for {symbol}")
        print(f"{'='*80}")
        
        # Fetch historical data (1 year daily)
        print(f"  📥 Fetching data for {symbol}...")
        stock_data = data_ingestion.fetch_stock_data(
            symbols=[symbol],
            period='1y',
            interval='1d'
        )
        
        if not stock_data or symbol not in stock_data:
            return False, f"❌ Failed to fetch data for {symbol}"
        
        data = stock_data[symbol]
        
        if len(data) < 100:
            return False, f"❌ Insufficient data for {symbol} (only {len(data)} rows)"
        
        print(f"  ✅ Fetched {len(data)} data points")
        
        # Train Random Forest model using RobustModelTrainer
        print(f"  🤖 Training Random Forest model...")
        results = trainer.run_comprehensive_training(
            data=data,
            symbol=symbol,
            algorithms=['random_forest'],
            model_type='classifier'
        )
        
        if results and 'models' in results:
            rf_result = results['models'].get('random_forest', {})
            if rf_result.get('status') == 'success':
                accuracy = rf_result.get('accuracy', 0)
                print(f"  ✅ Model trained successfully!")
                print(f"  📊 Accuracy: {accuracy:.2%}")
                return True, f"✅ {symbol}: Accuracy {accuracy:.2%}"
            else:
                return False, f"❌ Training failed for {symbol}: {rf_result.get('error', 'Unknown error')}"
        else:
            return False, f"❌ Training failed for {symbol}"
            
    except Exception as e:
        return False, f"❌ Error training {symbol}: {str(e)}"

def main():
    """Main batch training function"""
    print("\n" + "="*80)
    print("🚀 BATCH STOCK MODEL TRAINING")
    print("="*80)
    
    all_stocks = get_all_stocks()
    total_stocks = len(all_stocks)
    
    print(f"\n📋 Training models for {total_stocks} stocks across sectors:")
    for sector, stocks in STOCK_LIST.items():
        print(f"  {sector}: {', '.join(stocks)}")
    
    print(f"\n{'='*80}")
    
    # Initialize trainer and data ingestion
    trainer = RobustModelTrainer()
    data_ingestion = DataIngestion()
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'start_time': datetime.now()
    }
    
    # Train each stock
    for idx, symbol in enumerate(all_stocks, 1):
        print(f"\n\n[{idx}/{total_stocks}] Processing {symbol}...")
        
        success, message = train_stock_model(symbol, trainer, data_ingestion)
        
        if success:
            results['success'].append(message)
        else:
            results['failed'].append(message)
        
        # Small delay to avoid rate limiting
        if idx < total_stocks:
            time.sleep(1)
    
    # Summary
    results['end_time'] = datetime.now()
    duration = (results['end_time'] - results['start_time']).total_seconds()
    
    print("\n\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"✅ Successfully trained: {len(results['success'])}/{total_stocks}")
    print(f"❌ Failed: {len(results['failed'])}/{total_stocks}")
    
    if results['success']:
        print("\n✅ Successful Models:")
        for msg in results['success']:
            print(f"  {msg}")
    
    if results['failed']:
        print("\n❌ Failed Models:")
        for msg in results['failed']:
            print(f"  {msg}")
    
    print("\n" + "="*80)
    print(f"🎉 Batch training complete! {len(results['success'])} models ready for use.")
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with error code if more than 50% failed
    if len(results['failed']) > len(results['success']):
        sys.exit(1)
