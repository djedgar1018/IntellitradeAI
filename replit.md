# AI-Powered Trading Agent (IntelliTradeAI)

## Overview
This AI-powered trading agent provides real-time predictive signals across 39 cryptocurrencies, 108 stocks (all 11 GICS sectors), and 10 major ETFs. It leverages multiple machine learning models (Random Forest, XGBoost ensemble), explainable AI features, and comprehensive backtesting capabilities to generate trading signals. The system integrates real-time news intelligence and sophisticated signal fusion to provide actionable trading recommendations. It includes capabilities for options trading, automated execution with e-signature consent, blockchain integration, sentiment analysis, personalized trading plans based on risk tolerance, and SEC-compliant legal disclosures.

## Validated Model Performance (December 28, 2025)
**Prediction Target:** >4-5% significant price movement over 5-7 days
- **Stock Market (108 assets):** 85.2% average, 99.2% best (SO) - 98/108 stocks >= 70% (91%)
- **ETFs (10 assets):** 96.3% average, 98.8% best (DIA) - 10/10 >= 70% (100%)
- **Cryptocurrency (39 assets):** 54.7% average, 93.8% best (LEO) - 5/39 >= 70%
- **Overall (157 assets):** 78.4% average, 113/157 assets >= 70% (72%)
- **Top Stock Performers:** SO 99.2%, DUK 98.8%, PG 98.4%, TJX 98.4%, AVB 98.4%, MCD 98.0%
- **Top ETFs:** DIA 98.8%, XLV 98.0%, SPY 97.2%, XLF 97.6%
- **Top Crypto:** LEO 93.8%, TRX 86.0%, BTC-USD 80.3%, BNB 75.6%, TON 74.4%
- Detailed results: `model_results/december_2025_results.json`
- Validation report: `model_results/VALIDATION_REPORT.md`

## Recent Changes (December 2025)
- **Model Training Optimization (Dec 28)**: Created comprehensive training pipeline with validated accuracy
  - Implemented 70 technical indicators with feature selection
  - Added SMOTE class balancing and temporal train/test splits
  - Final ensemble: RandomForest + XGBoost voting classifier
  - Walk-forward validation for proper time-series evaluation
- **IEEE Paper Revisions (Dec 30)**: Updated TEX file to match final PDF submission
  - Author: Danario Edgar II, Prairie View A&M University (dedgar1@pvamu.edu)
  - Abstract: Uses "novel tri-signal fusion", "Baysian hyperparameter optimization"
  - Language: Changed to first person ("My contribution", "My tool", "My research")
  - Models: "Random Forest, XGBoost, Random Forest+XGBoost, and voting ensemble classifiers"
  - Accuracy: "85.2% for standalone ML accuracy by 5.4 percentage points (9.9% relative improvement)"
  - Citations updated: Cheng, Yanxi, Christine et al. references
  - Simplified structure matching PDF layout
  - Paper now 292 lines, 27 citations, 6 tables, 4 figures
- **GitHub Preparation**: Created README.md, requirements.txt, LICENSE, .gitignore
- Added SEC/FINRA legal compliance module with risk disclosures and e-signature authorization
- Created hover-based tooltip definitions (3-second delay) replacing standalone dictionary tab
- Implemented user onboarding survey with 5 risk tolerance levels
- Expanded crypto coverage to top 100 coins with 12 sector categorizations
- Added all 11 GICS stock sectors with industries and 30+ ETF indices
- Created personalized trading plan system from Conservative to Speculative
- Added database tables for user profiles and e-signature records
- Added sector & ETF rankings table with AI scores in trading plan
- Added popup charts with optimal levels (support/resistance/entry/exit) for assets
- Implemented price alerts functionality for recommended assets
- Enhanced options trading plan with tier-specific call/put suggestions
- Fixed options analysis to suggest both calls and puts for all strategies

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
A Streamlit-based dashboard provides an interactive web UI with real-time data visualization, a multi-page layout for trading, backtesting, model comparison, analytics, and options analysis, all designed responsively with a wide layout and expandable sidebar. The UI includes comprehensive data visualization with 6 charts, interactive price charts with visual key levels, and actionable price levels for HOLD signals.

### Backend Architecture
The backend features a modular design separating data fetching, model training, backtesting, and analysis. It includes a centralized configuration system, a standardized model pipeline supporting multiple ML models, and a robust data processing pipeline for cleaning and technical indicator calculation. The system also incorporates a "TRI-SIGNAL FUSION ENGINE" that combines ML model predictions, pattern recognition, and news intelligence with weighted voting and smart conflict resolution.

### Data Storage Solutions
File-based storage is used for caching cryptocurrency and stock data in JSON format, with in-memory processing using pandas DataFrames. Model persistence is handled via Joblib serialization, and Streamlit session state maintains application state. Database integration with PostgreSQL is used for trades, positions, portfolio, trade alerts, options chains, and crypto wallets.

### Key Components
- **Data Ingestion Layer**: Integrates CoinMarketCap API (cryptocurrency) and Yahoo Finance (stocks) with JSON-based caching, dynamic top coin discovery, and comprehensive data validation.
- **Machine Learning Models**: Employs RandomForest + XGBoost voting ensemble with 70 technical indicators and SMOTE class balancing.
- **Technical Analysis Engine**: Calculates RSI, MACD, Bollinger Bands, EMA, and performs automated feature engineering and cross-market correlation analysis.
- **Explainability and Transparency**: Integrates SHAP for model interpretability, logs trading decisions, and visualizes predictions and explanations.
- **Backtesting Engine**: Provides custom backtesting with configurable parameters, performance metrics (Sharpe ratio, max drawdown), and built-in risk management (stop-loss, take-profit).
- **Options Trading**: Includes an Options Chain Data Fetcher with real-time calls/puts data, Greeks, and implied volatility.
- **Automated Execution**: Features a Trading Mode Manager for manual or automatic AI execution, and a Trade Executor supporting stocks, options, and crypto with paper trading capabilities.
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