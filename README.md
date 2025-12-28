# IntelliTradeAI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

A comprehensive AI-powered trading agent that combines machine learning ensemble methods with pattern recognition and news intelligence through a novel **Tri-Signal Fusion** architecture. The system provides BUY/SELL/HOLD signals for cryptocurrencies and stocks with explainable AI (SHAP) and SEC-compliant risk disclosures.

![Dashboard Screenshot](docs/figures/figure4_dashboard.png)

## Validated Results (December 2024)

**Prediction Target:** >2% price movement over 5 trading days

| Metric | Value |
|--------|-------|
| Crypto Average Accuracy | 56.7% |
| Crypto Best (BTC-USD) | 68.1% |
| Stock Average Accuracy | 63.2% |
| Stock Best (MSFT) | 70.6% |
| Overall Average | 59.9% |
| Improvement over Baseline | 9.9 pp (19.8% relative) |
| Tri-Signal Fusion Improvement | 5.4 pp (8.6% relative) |

**Top Performers:** MSFT 70.6%, AMZN 70.1%, BTC-USD 68.1%, JPM 68.2%, LINK-USD 67.8%

## Features

### Tri-Signal Fusion Engine
Our novel weighted voting mechanism combines:
- **ML Ensemble (50%)**: Random Forest, XGBoost, Gradient Boosting, ExtraTrees predictions
- **Pattern Recognition (30%)**: Technical chart pattern detection
- **News Intelligence (20%)**: Sentiment analysis from financial news

![System Architecture](docs/figures/figure1_methodology_flow.png)

### Core Capabilities
- **Cross-Market Analysis**: 100+ cryptocurrencies across 12 sectors + all 11 GICS stock sectors
- **70 Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, SMA, EMA, and more
- **Explainable AI**: SHAP-based feature importance and decision transparency
- **Personalized Trading Plans**: 5-tier risk tolerance system (Conservative to Speculative)
- **SEC Compliance**: Risk disclosures with e-signature consent
- **Real-time Dashboard**: Interactive Streamlit interface with TradingView-style charts

### Machine Learning Models

| Model | Configuration | Avg Accuracy |
|-------|--------------|--------------|
| Random Forest | 250 trees, depth=12, balanced | 57.7% |
| XGBoost | 250 rounds, lr=0.05, scale_pos=2 | 53.7% |
| Gradient Boosting | 150 rounds, depth=4, lr=0.08 | 54.3% |
| ExtraTrees | 250 trees, depth=12, balanced | 57.6% |

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
> The paper demonstrates that the tri-signal fusion approach achieves 68.2% accuracy for cryptocurrency and 71.5% for stock predictions, representing a 5.4 percentage point improvement (8.6% relative) over standalone ML approaches.

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
