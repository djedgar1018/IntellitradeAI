# AI-Powered Trading Agent (IntelliTradeAI)

## Overview
IntelliTradeAI is an AI-powered trading agent providing real-time predictive signals across 141 cryptocurrencies, 108 stocks, and 10 major ETFs. It leverages multiple machine learning models (Random Forest, XGBoost, LSTM, Transformer), explainable AI, and comprehensive backtesting to generate trading signals. The system integrates real-time news intelligence, social sentiment analysis, on-chain metrics, and sophisticated tri-signal fusion for actionable recommendations. Key capabilities include options trading, automated execution, blockchain integration, sentiment analysis, personalized trading plans based on risk tolerance, and SEC-compliant legal disclosures. The project aims to deliver highly accurate, explainable, and adaptable trading intelligence to users.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
A Streamlit-based interactive web UI provides a multi-page layout for trading, backtesting, model comparison, analytics, and options analysis. It features real-time data visualization with 6 charts, interactive price charts with visual key levels, and actionable price levels for HOLD signals, all designed responsively with a wide layout and expandable sidebar.

### Backend Architecture
The backend is modular, separating data fetching, model training, backtesting, and analysis. It includes a centralized configuration system, a standardized model pipeline supporting multiple ML models, and a robust data processing pipeline. A "TRI-SIGNAL FUSION ENGINE" combines ML predictions, pattern recognition, and news intelligence using weighted voting and smart conflict resolution.

### Data Storage Solutions
File-based storage is used for caching cryptocurrency and stock data in JSON format, with in-memory processing via pandas DataFrames. Model persistence utilizes Joblib serialization, and Streamlit session state maintains application state. PostgreSQL is integrated for trades, positions, portfolio, trade alerts, options chains, and crypto wallets.

### Key Components
- **Data Ingestion**: Integrates CoinMarketCap API and Yahoo Finance with JSON caching, dynamic top coin discovery, and comprehensive data validation for 141 crypto symbols and stocks.
- **Machine Learning Models**: Employs a RandomForest + XGBoost voting ensemble with 70+ technical indicators, 47 volatility-aware features, and SMOTE class balancing. LSTM and Transformer models are available for sequence prediction.
- **Volatility-Aware Training**: Features adaptive threshold training based on asset volatility classes and sector-specific model configurations.
- **Alternative Data Sources**: Incorporates on-chain metrics (whale activity, exchange flows) and social sentiment (Fear & Greed Index, price/volume sentiment).
- **Technical Analysis Engine**: Calculates key indicators like RSI, MACD, Bollinger Bands, EMA, and performs automated feature engineering and cross-market correlation analysis.
- **Explainability**: Integrates SHAP for model interpretability, logging trading decisions, and visualizing predictions.
- **Backtesting Engine**: Provides custom backtesting with configurable parameters, performance metrics (Sharpe ratio, max drawdown), and built-in risk management.
- **Options Trading**: Includes an Options Chain Data Fetcher for real-time calls/puts data, Greeks, and implied volatility.
- **Automated Execution**: Features a Trading Mode Manager for manual or automatic AI execution and a Trade Executor supporting stocks, options, and crypto with paper trading.
- **Blockchain Integration**: Utilizes a Blockchain Wallet Manager (Web3.py) for crypto transactions and wallet tracking.
- **Sentiment Analysis**: Incorporates Twitter/X sentiment analysis and a Fear & Greed Index display.
- **Trade Logging & P&L**: Tracks trades, positions, and performance analytics.

## External Dependencies

### APIs and Data Sources
- **CoinMarketCap API**: Cryptocurrency market data.
- **Yahoo Finance**: Stock market data and RSS news feed.

### Machine Learning Libraries
- **TensorFlow/Keras**: Deep learning framework.
- **Scikit-learn**: ML algorithms and preprocessing.
- **XGBoost**: Gradient boosting framework.
- **SHAP**: Model explainability.

### Visualization and UI
- **Streamlit**: Web application framework.
- **Plotly**: Interactive charting.
- **Pandas**: Data manipulation.

### Configuration Management
- **Python-dotenv**: Environment variable management.
- **JSON**: Data serialization and caching.

### Database
- **PostgreSQL**: For storing trade-related data, positions, portfolio, alerts, options chains, and crypto wallets.

### Other Integrations
- **Web3.py**: For blockchain wallet management.
- **Discord Integration**: Analyze trading conversations from Discord servers (Honey Drip Network, TJRTrades)
- **Email Newsletter Integration**: Extract trading signals from TLDR AI, Barchart, Investing.com, Webull newsletters via Gmail IMAP

## Recent Changes (January 2026)
- **Email Newsletter Integration (Jan 7)**: Extract trading signals from email newsletters
  - Gmail IMAP client with App Password authentication
  - Newsletter parser for TLDR AI, Barchart, Investing.com, Webull
  - Symbol intelligence with bullish/bearish bias detection
  - Files: email_integration/ (imap_client.py, newsletter_parser.py, email_service.py)

- **Discord Integration (Jan 7)**: Analyze trading conversations from Discord
  - Trade message parser for crypto, stocks, options
  - Trade history analyzer for pattern detection
  - Database persistence for learned patterns
  - Files: discord_integration/ (client.py, trade_parser.py, trade_analyzer.py, discord_service.py)