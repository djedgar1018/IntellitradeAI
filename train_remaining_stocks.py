"""Train the remaining top stocks"""
import time
from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion

# Remaining stocks to train
REMAINING = ['V', 'MA', 'WMT', 'JNJ', 'XOM', 'AMD', 'NFLX']

trainer = RobustModelTrainer()
ing = DataIngestion()

print(f"Training {len(REMAINING)} remaining stocks...")
success = 0

for i, symbol in enumerate(REMAINING, 1):
    try:
        print(f"\n[{i}/{len(REMAINING)}] {symbol}...")
        data = ing.fetch_stock_data([symbol], period='1y', interval='1d')
        if data and symbol in data:
            result = trainer.run_comprehensive_training(
                data=data[symbol],
                symbol=symbol,
                algorithms=['random_forest'],
                optimize_hyperparams=True
            )
            if result:
                print(f"✅ {symbol} complete")
                success += 1
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {symbol} error: {e}")

print(f"\n✅ Trained {success}/{len(REMAINING)} stocks")
