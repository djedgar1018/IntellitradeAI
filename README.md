# IntelliTradeAI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

A comprehensive AI-powered trading agent that combines machine learning ensemble methods with pattern recognition and news intelligence through a novel **Tri-Signal Fusion** architecture. The system provides BUY/SELL/HOLD signals for cryptocurrencies and stocks with explainable AI (SHAP) and SEC-compliant risk disclosures.

![Dashboard Screenshot](docs/figures/figure4_dashboard.png)

## Validated Results (December 2025)

**Prediction Target:** Significant price movement with adaptive thresholds

| Asset Class | Count | Average | Best | >= 70% |
|-------------|-------|---------|------|--------|
| **Stocks** | 108 | **85.2%** | 99.2% (SO) | **98/108 (91%)** |
| **ETFs** | 10 | **96.3%** | 98.8% (DIA) | **10/10 (100%)** |
| **Top 10 Crypto** | 10 | **72.9%** | 92.4% (BTC) | **5/10 (50%)** |
| **Overall** | **259** | **~82%** | - | **113+ (Stocks/ETFs)** |

### Top 10 Cryptocurrency Accuracy (Volatility-Aware Training)

| Coin | Accuracy | Threshold | Horizon |
|------|----------|-----------|---------|
| **BTC** | **92.4%** | 6% move | 7 days |
| **XRP** | **88.1%** | 6% move | 5 days |
| **DOGE** | **76.7%** | 8% move | 5 days |
| **ETH** | **71.4%** | 6% move | 7 days |
| **SOL** | **71.0%** | 8% move | 5 days |
| TRX | 69.5% | 5% move | 7 days |
| BNB | 68.6% | 6% move | 7 days |
| ADA | 67.1% | 8% move | 7 days |
| SHIB | 63.8% | 8% move | 5 days |
| AVAX | 60.5% | 8% move | 5 days |

**Top Stock Performers:** SO 99.2%, DUK 98.8%, PG 98.4%, TJX 98.4%, AVB 98.4%, MCD 98.0%

**Crypto Improvement:** 72.9% average (up from 54.7% baseline - **33% improvement**)

**Coverage:** 141 cryptocurrencies (14 sectors) + 108 stocks (11 GICS sectors) + 10 major ETFs

## Features

### Tri-Signal Fusion Engine
Our novel weighted voting mechanism combines:
- **ML Ensemble (50%)**: Random Forest + XGBoost voting ensemble predictions
- **Pattern Recognition (30%)**: Technical chart pattern detection
- **News Intelligence (20%)**: Sentiment analysis from financial news

![System Architecture](docs/figures/figure1_methodology_flow.png)

### Core Capabilities
- **Cross-Market Analysis**: 141 cryptocurrencies (14 sectors), 108 stocks across all 11 GICS sectors, and 10 major ETFs
- **70 Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, SMA, EMA, and more
- **Explainable AI**: SHAP-based feature importance and decision transparency
- **Personalized Trading Plans**: 5-tier risk tolerance system (Conservative to Speculative)
- **SEC Compliance**: Risk disclosures with e-signature consent
- **Real-time Dashboard**: Interactive Streamlit interface with TradingView-style charts

### Machine Learning Models

| Model | Configuration | Role |
|-------|--------------|------|
| Random Forest | 150 trees, depth=10, balanced | Primary classifier |
| XGBoost | 150 rounds, lr=0.05, scale_pos=3 | Secondary classifier |
| Voting Ensemble | Soft voting combination | Final prediction (78.4% avg) |

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL database
- CoinMarketCap API key

### Setup

```bash
# Clone the repository
git clone https://github.com/intellitradeai/intellitradeai.git
cd intellitradeai

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export COINMARKETCAP_API_KEY="your_api_key"
export DATABASE_URL="postgresql://user:pass@localhost:5432/intellitradeai"

# Run the application
streamlit run app/enhanced_dashboard.py --server.port 5000
```

## Project Structure

```
intellitradeai/
├── app/                    # Streamlit dashboard application
│   └── enhanced_dashboard.py
├── models/                 # ML model implementations
│   ├── lstm_model.py
│   ├── random_forest_model.py
│   └── xgboost_model.py
├── trading/                # Trading logic and execution
├── backtest/               # Backtesting framework
├── sentiment/              # News sentiment analysis
├── compliance/             # SEC compliance and disclosures
├── database/               # Database schemas and migrations
├── docs/                   # Documentation and IEEE paper
│   ├── figures/            # Paper figures (fig1-4.png)
│   └── IntelliTradeAI_IEEE_Paper.tex
├── config/                 # Configuration files
├── utils/                  # Utility functions
└── tests/                  # Test suite
```

## Usage

### Running the Dashboard

```bash
streamlit run app/enhanced_dashboard.py --server.port 5000
```

### Training Models

```bash
python train_top10_models.py
```

### Running Backtests

```bash
python -m backtest.backtest_engine --symbol BTC-USD --period 2y
```

## API Endpoints

The FastAPI backend provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signals/{symbol}` | GET | Get trading signal for asset |
| `/api/backtest` | POST | Run backtest simulation |
| `/api/portfolio` | GET | Get portfolio summary |
| `/api/alerts` | POST | Create price alert |

## Academic Paper

This research is documented in an IEEE conference paper:

> **IntelliTradeAI: A Tri-Signal Fusion Framework for Explainable AI-Powered Financial Market Prediction**
>
> The paper demonstrates validated prediction accuracy of 85.2% for stocks (108 assets, 91% >= 70%), 96.3% for ETFs (10 assets, 100% >= 70%), and 54.7% for cryptocurrencies (39 assets) - representing an overall 78.4% average accuracy across 157 tested assets.

Paper files are available in `docs/`:
- `IntelliTradeAI_IEEE_Paper.tex` - LaTeX source
- `figures/fig1.png` - System architecture
- `figures/fig2.png` - Training loss curves
- `figures/fig3.png` - Backtest comparison
- `figures/fig4.png` - Dashboard screenshot

## Risk Disclosure

**Important**: This software is for educational and research purposes only. 

- Past performance does not guarantee future results
- Cryptocurrency and stock trading involves substantial risk of loss
- This is not financial advice - consult a qualified financial advisor
- Users must acknowledge risk disclosures before using automated features

## Data Sources

- **CoinMarketCap API**: Real-time cryptocurrency data
- **Yahoo Finance**: Stock market data and historical prices
- **News APIs**: Financial news for sentiment analysis

## Technologies

- **Frontend**: Streamlit, Plotly
- **Backend**: FastAPI, Python 3.11
- **ML/AI**: TensorFlow/Keras, Scikit-learn, XGBoost, SHAP
- **Database**: PostgreSQL
- **Blockchain**: Web3.py

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{intellitradeai2024,
  title={IntelliTradeAI: A Tri-Signal Fusion Framework for Explainable AI-Powered Financial Market Prediction},
  author={Author Name},
  booktitle={IEEE SoutheastCon 2026},
  year={2026}
}
```

## Contact

For questions or support, please open an issue on GitHub.
