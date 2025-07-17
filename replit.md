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

- **2025-01-17**: Added dedicated `data/data_ingestion.py` module for API calls
- **2025-01-17**: Created `models/model_trainer.py` for centralized ML model training
- **2025-01-17**: Implemented comprehensive `.gitignore` file to protect sensitive data
- **2025-01-17**: Fixed TensorFlow compatibility issues by making LSTM model optional
- **2025-01-17**: Resolved technical indicators calculation errors
- **2025-01-17**: Enhanced error handling for cache validity checks

## File Structure Updates

### Data Layer
- `data/data_ingestion.py`: Centralized API calls for cryptocurrency and stock data
- `data/crypto_data.json`: Cached cryptocurrency data
- `data/stock_data.json`: Cached stock market data

### Models Layer
- `models/model_trainer.py`: Centralized ML model training logic
- `models/lstm_model.py`: LSTM neural network (optional, requires TensorFlow)
- `models/random_forest_model.py`: Random Forest classifier/regressor
- `models/xgboost_model.py`: XGBoost classifier/regressor
- `models/model_comparison.py`: Framework for comparing model performance

The system is designed to be extensible and maintainable, with clear separation of concerns and standardized interfaces between components. The configuration-driven approach allows for easy customization of model parameters, API endpoints, and system behavior.