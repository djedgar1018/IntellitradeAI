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
    
    # Trading Parameters - V22 Optimized for 5x/10x Growth
    TRADING_CONFIG = {
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
        "risk_tolerance": 0.28,  # V22: 28% base risk for aggressive growth
        "max_position_size": 0.85,  # V22: Up to 85% in single position
        "stop_loss": 0.0035,  # V22: 0.35% tight stops for scalping
        "take_profit": 0.10,  # V22: 10% profit targets
        "max_hold_days": 2,  # V22: Short holds for scalping
        "win_streak_multiplier": 3.8,  # V22: Compound on win streaks
        "pyramid_max": 5,  # V22: Up to 5 add-ons per position
        "pyramid_add_percent": 0.75,  # V22: 75% position add each pyramid
        "volatility_bonus_max": 1.35  # V22: Up to 35% bonus on high volatility
    }
    
    # Asset-specific V22 configurations
    # Note: Stop/target ranges vary by asset volatility:
    # - Stocks: 0.35% stop (moderate volatility)
    # - Crypto: 0.50% stop (high volatility)
    # - Forex: 0.18% stop (low volatility, tight spreads)
    # - Options: 0.25% stop on underlying (leveraged exposure)
    V22_SCALP_CONFIG = {
        "stocks": {
            "max_positions": 8,
            "base_risk_pct": 28.0,
            "max_position_pct": 82.0,
            "stop_loss_pct": 0.35,  # 0.35% stop for stocks
            "target_pct": 9.5,      # 9.5% target
            "max_hold_days": 2
        },
        "crypto": {
            "max_positions": 6,
            "base_risk_pct": 35.0,
            "max_position_pct": 88.0,
            "stop_loss_pct": 0.50,  # 0.50% stop (higher volatility)
            "target_pct": 14.0,     # 14% target
            "max_hold_days": 1
        },
        "forex": {
            "max_positions": 6,
            "base_risk_pct": 30.0,
            "max_position_pct": 85.0,
            "stop_loss_pct": 0.18,  # 0.18% stop (tight forex spreads)
            "target_pct": 6.5,      # 6.5% target
            "max_hold_days": 2
        },
        "options": {
            "max_positions": 8,
            "base_risk_pct": 32.0,
            "max_position_pct": 85.0,
            "stop_loss_pct": 0.25,  # 0.25% on underlying
            "target_pct": 7.5,      # 7.5% on underlying
            "max_hold_days": 2
        }
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
