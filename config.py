"""
Configuration module for AI Trading Agent
Centralized configuration management for API routes and model parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the AI Trading Agent"""
    
    # API Configuration
    COINMARKETCAP_API_KEY = os.getenv("CMC_API_KEY", "default_key")
    COINMARKETCAP_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
    
    # Yahoo Finance doesn't require API key
    YAHOO_FINANCE_TIMEOUT = 30
    
    # Model Parameters
    LSTM_CONFIG = {
        "sequence_length": 60,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "dropout_rate": 0.2,
        "units": 50
    }
    
    RANDOM_FOREST_CONFIG = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    
    XGBOOST_CONFIG = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    
    # Technical Indicators Parameters
    INDICATOR_CONFIG = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "ema_periods": [12, 26, 50, 200]
    }
    
    # Trading Parameters
    TRADING_CONFIG = {
        "buy_threshold": 0.7,
        "sell_threshold": 0.3,
        "risk_tolerance": 0.02,
        "max_position_size": 0.1,
        "stop_loss": 0.05,
        "take_profit": 0.15
    }
    
    # Data Configuration
    DATA_CONFIG = {
        "crypto_symbols": ["BTC", "ETH", "ADA", "SOL", "MATIC"],
        "stock_symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        "data_period": "1y",
        "data_interval": "1d",
        "cache_duration": 3600  # seconds
    }
    
    # Backtesting Configuration
    BACKTEST_CONFIG = {
        "initial_capital": 10000,
        "commission": 0.001,
        "slippage": 0.001,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31"
    }
    
    # File Paths
    PATHS = {
        "crypto_data": "data/crypto_data.json",
        "stock_data": "data/stock_data.json",
        "model_cache": "models/cache/",
        "backtest_results": "backtest/results/"
    }
    
    # UI Configuration
    UI_CONFIG = {
        "refresh_interval": 300,  # seconds
        "max_chart_points": 1000,
        "default_timeframe": "1d"
    }

# Create global config instance
config = Config()
