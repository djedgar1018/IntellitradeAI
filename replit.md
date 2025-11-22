# AI-Powered Trading Agent

## Overview

This is a comprehensive AI-powered trading agent that provides real-time predictive signals across cryptocurrency and stock markets. The system uses multiple machine learning models (LSTM, Random Forest, XGBoost) to generate trading signals with explainable AI features and comprehensive backtesting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Interface**: Streamlit-based dashboard providing an interactive web UI
- **Real-time Updates**: Live data visualization with configurable refresh intervals
- **Multi-page Layout**: Organized into sections for trading, backtesting, model comparison, and analytics
- **Responsive Design**: Wide layout with expandable sidebar for navigation

### Backend Architecture
- **Modular Design**: Separated into distinct modules for data fetching, model training, backtesting, and analysis
- **Configuration Management**: Centralized configuration system for API keys, model parameters, and system settings
- **Model Pipeline**: Supports multiple ML models with standardized interfaces for training and prediction
- **Data Processing**: Comprehensive data cleaning and technical indicator calculation pipeline

### Data Storage Solutions
- **File-based Storage**: JSON files for caching cryptocurrency and stock data
- **In-memory Processing**: Real-time data processing using pandas DataFrames
- **Model Persistence**: Joblib-based model serialization for trained models
- **Session State**: Streamlit session state for maintaining application state

## Key Components

### Data Ingestion Layer
- **CoinMarketCap Integration**: Real-time cryptocurrency data fetching with API key authentication
- **Yahoo Finance Integration**: Stock market data retrieval without API requirements
- **Data Caching**: JSON-based caching system with configurable cache duration
- **Data Validation**: Comprehensive OHLCV data cleaning and validation

### Machine Learning Models
- **LSTM Neural Network**: Time series prediction using TensorFlow/Keras with configurable architecture
- **Random Forest**: Ensemble method for both classification and regression tasks
- **XGBoost**: Gradient boosting framework with optimized hyperparameters
- **Model Comparison**: Framework for evaluating multiple models side-by-side

### Technical Analysis Engine
- **Technical Indicators**: RSI, MACD, Bollinger Bands, EMA calculations
- **Feature Engineering**: Automated feature creation from OHLCV data
- **Cross-market Analysis**: Correlation analysis between different asset classes

### Explainability and Transparency
- **SHAP Integration**: Model interpretability using SHAP values
- **Decision Logging**: Comprehensive logging of trading decisions and rationale
- **Visualization**: Interactive charts showing model predictions and explanations

### Backtesting Engine
- **Custom Backtesting**: Comprehensive backtesting with configurable parameters
- **Performance Metrics**: Detailed performance analysis including Sharpe ratio, maximum drawdown
- **Risk Management**: Built-in stop-loss and take-profit mechanisms

## Data Flow

1. **Data Acquisition**: External APIs (CoinMarketCap, Yahoo Finance) → Data Fetcher
2. **Data Processing**: Raw data → Data Cleaner → Technical Indicators → Feature Engineering
3. **Model Training**: Processed data → ML Models → Trained models (cached)
4. **Prediction**: Current data → Trained models → Trading signals
5. **Backtesting**: Historical data → Backtesting Engine → Performance metrics
6. **Visualization**: Results → Streamlit Dashboard → User interface

## External Dependencies

### APIs and Data Sources
- **CoinMarketCap API**: Cryptocurrency market data (requires API key)
- **Yahoo Finance**: Stock market data (free, no API key required)

### Machine Learning Libraries
- **TensorFlow/Keras**: Deep learning framework for LSTM models
- **Scikit-learn**: Traditional ML algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **SHAP**: Model explainability and interpretability

### Visualization and UI
- **Streamlit**: Web application framework for the dashboard
- **Plotly**: Interactive charting and visualization
- **Pandas**: Data manipulation and analysis

### Configuration Management
- **Python-dotenv**: Environment variable management for API keys
- **JSON**: Data serialization and caching

## Deployment Strategy

### Local Development
- **Streamlit Application**: Run locally using `streamlit run main.py`
- **Environment Setup**: Requires Python environment with dependencies from requirements (not included in repository)
- **API Configuration**: Requires CoinMarketCap API key in environment variables

### Production Considerations
- **Scalability**: Modular architecture supports horizontal scaling
- **Caching**: File-based caching reduces API calls and improves performance
- **Error Handling**: Comprehensive error handling throughout the application
- **Resource Management**: Configurable timeouts and rate limiting for API calls

### Database Integration Notes
- **Current State**: Uses file-based storage with JSON files
- **Future Enhancement**: The architecture supports easy migration to database systems
- **Data Persistence**: Model states and application data are maintained in session state and files

## Recent Changes

- **2025-11-22**: 🎯 **CRITICAL FIX: Signal Fusion Engine** - Resolved conflicting signals issue where ML predictor and pattern recognizer gave contradictory recommendations (e.g., BUY vs SELL)
- **2025-11-22**: Created `SignalFusionEngine` that intelligently combines ML predictions and chart patterns with conflict resolution logic
- **2025-11-22**: Implemented smart conflict rules: defaults to HOLD when both systems have high confidence (>65%) but disagree
- **2025-11-22**: Enhanced dashboard UI to show unified signal with both ML insight and Pattern insight side-by-side for transparency
- **2025-11-22**: Added visual conflict warnings (red border) when AI systems disagree to protect users from risky trades
- **2025-11-22**: 📊 **NEW: Actionable Price Levels for HOLD Signals** - Created `PriceLevelAnalyzer` that calculates 3 key support/resistance levels
- **2025-11-22**: HOLD signals now show specific price targets with BUY/SELL recommendations at each level (e.g., "BUY at $49.00 support", "SELL at $55.00 resistance")
- **2025-11-22**: Price levels use technical analysis (swing highs/lows, moving averages, round numbers) with confidence scores
- **2025-11-19**: 🚀 **Major Enhancement: Top 10 Coins Support** - System now dynamically fetches and supports top 10 cryptocurrencies from CoinMarketCap
- **2025-11-19**: Created `TopCoinsManager` for dynamic coin discovery with 1-hour caching (100% success rate)
- **2025-11-19**: Built `EnhancedCryptoFetcher` with multi-coin support, portfolio analytics, and robust error handling
- **2025-11-19**: Added comprehensive Yahoo Finance symbol mapping for 30+ cryptocurrencies
- **2025-11-19**: Implemented 3-level fallback system (CoinMarketCap → Cache → Defaults) for maximum reliability
- **2025-11-19**: Tested successfully with all 10 coins: BTC, ETH, USDT, XRP, BNB, SOL, USDC, TRX, DOGE, ADA (1,850 data points)
- **2025-11-19**: Created complete ML documentation suite with 8 professional visualizations (confusion matrix, ROC curves, feature importance, etc.)
- **2025-11-19**: Documented complete training methodology: dataset specs, feature engineering (70+ features), train/test split, hyperparameter tuning
- **2025-11-14**: Integrated CoinMarketCap API securely via Replit Secrets; implemented hybrid data fetching (Yahoo Finance + CoinMarketCap)
- **2025-07-25**: Built FastAPI web service with comprehensive REST endpoints for model operations
- **2025-07-25**: Created Streamlit dashboard with menu-driven interface for easy user interaction  
- **2025-07-25**: Implemented full-stack architecture with API backend and web frontend
- **2025-07-25**: Added FastAPI endpoints: /, /retrain, /data, /predict, /models, /health
- **2025-07-25**: Built interactive Streamlit UI with Overview, Data Fetching, Model Training, and Prediction sections
- **2025-07-25**: Configured dual workflow setup: FastAPI on port 8000, Streamlit on port 5000
- **2025-07-25**: Enhanced model trainer with cached model detection and prediction capabilities
- **2025-07-25**: Successfully tested end-to-end functionality: data fetching → model training → predictions

## File Structure Updates

### Data Layer
- `data/data_ingestion.py`: Centralized API calls for cryptocurrency and stock data
- `data/crypto_data_fetcher.py`: Original crypto data fetcher (BTC, ETH, LTC)
- `data/top_coins_manager.py`: **NEW** - Dynamic top 10 coins from CoinMarketCap with caching
- `data/enhanced_crypto_fetcher.py`: **NEW** - Multi-coin fetcher with portfolio analytics (supports top N coins)
- `data/crypto_data.json`: Cached cryptocurrency data
- `data/stock_data.json`: Cached stock market data
- `data/top_coins_cache.json`: **NEW** - Cached top 10 list (1-hour TTL)
- `data/crypto_top10_cache.json`: **NEW** - Cached OHLCV data for top 10 coins

### Models Layer
- `models/model_trainer.py`: Centralized ML model training logic
- `models/lstm_model.py`: LSTM neural network (optional, requires TensorFlow)
- `models/random_forest_model.py`: Random Forest classifier/regressor
- `models/xgboost_model.py`: XGBoost classifier/regressor
- `models/model_comparison.py`: Framework for comparing model performance

### AI Advisor Layer
- `ai_advisor/signal_fusion_engine.py`: **NEW** - Intelligent signal fusion with conflict resolution
- `ai_advisor/price_level_analyzer.py`: **NEW** - Support/resistance analysis for HOLD signals
- `ai_advisor/ml_predictor.py`: ML-based prediction engine
- `ai_vision/chart_pattern_recognition.py`: Technical pattern recognition engine

The system is designed to be extensible and maintainable, with clear separation of concerns and standardized interfaces between components. The configuration-driven approach allows for easy customization of model parameters, API endpoints, and system behavior.