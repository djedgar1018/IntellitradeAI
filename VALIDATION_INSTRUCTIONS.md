# IntelliTradeAI Model Validation Instructions

This document provides step-by-step instructions to reproduce and validate the accuracy results reported in the IEEE SoutheastCon 2026 paper.

## Quick Start

```bash
# Run the main validation script
python train_final_publication.py
```

Results will be saved to: `model_results/final_publication_results.json`

## Prerequisites

### Required Python Packages
```bash
pip install pandas numpy scikit-learn xgboost yfinance imbalanced-learn
```

### Environment
- Python 3.11+
- Internet connection (for downloading market data from Yahoo Finance)

## Validation Process

### Step 1: Run the Training Script

```bash
python train_final_publication.py
```

This script will:
1. Download 5 years of historical data for 20 assets (10 crypto, 10 stocks)
2. Calculate 70+ technical indicators
3. Create prediction targets (>2% price movement over 5 days)
4. Train 4 models (RandomForest, XGBoost, GradientBoosting, ExtraTrees)
5. Evaluate using temporal 80/20 train/test split with SMOTE balancing
6. Report accuracy metrics for each asset

### Step 2: Review Results

The script outputs:
- Individual asset accuracy with best model
- Average accuracy by asset class (crypto vs stocks)
- Model comparison across all assets
- JSON file with detailed results

### Step 3: Verify Against Published Metrics

**Expected Results (December 2025):**

| Asset Class | Average Accuracy | Best Individual | >= 70% |
|-------------|------------------|-----------------|--------|
| Stock Market (10) | **81.5%** | **92.1% (V)** | **10/10** |
| Cryptocurrency (10) | 52.4% | 80.3% (BTC-USD) | 1/10 |
| Overall (20) | 66.9% | - | 11/20 |

**Stock Results (ALL >= 70%):**
- V: 92.1%
- JPM: 89.6%
- MSFT: 87.6%
- AAPL: 83.8%
- META: 81.3%
- GOOGL: 79.7%
- WMT: 78.8%
- AMZN: 75.9%
- NVDA: 73.4%
- TSLA: 72.2%

**Cryptocurrency Results:**
- BTC-USD: 80.3% (>= 70%)
- XRP-USD: 67.7%
- SOL-USD: 66.3%
- ETH-USD: 59.6%
- ADA-USD: 49.4%
- LINK-USD: 46.3%
- DOGE-USD: 43.0%
- MATIC-USD: 39.7%
- AVAX-USD: 36.2%
- DOT-USD: 35.7%

## Methodology Details

### Prediction Target
- **Definition:** Binary classification - will price increase by >2% within 5 trading days?
- **Class Balance:** Approximately 35% positive (significant moves)
- **Baseline:** 50% random accuracy

### Data Split
- **Training:** First 80% of data (chronologically)
- **Testing:** Last 20% of data
- **No data leakage:** Future data never seen during training

### Class Balancing
- **Method:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Purpose:** Address class imbalance (~35% positive samples)

### Feature Engineering
70+ technical indicators including:
- Moving Averages (SMA, EMA at 5, 10, 20, 50, 100, 200 periods)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume (OBV, Volume Ratio)
- Statistical (Skewness, Kurtosis)

### Model Configurations

| Model | Key Parameters |
|-------|----------------|
| RandomForest | 250 trees, depth=12, class_weight='balanced' |
| XGBoost | 250 rounds, lr=0.05, scale_pos_weight=2 |
| GradientBoosting | 150 rounds, depth=4, lr=0.08 |
| ExtraTrees | 250 trees, depth=12, class_weight='balanced' |

## Reproducing Specific Experiments

### Test Different Thresholds

Edit `train_final_publication.py` and change the threshold parameter:

```python
# For >3% moves (higher accuracy, fewer signals)
r = train_asset(s, 'crypto', horizon=5, threshold=3.0)

# For >1% moves (lower accuracy, more signals)  
r = train_asset(s, 'crypto', horizon=5, threshold=1.0)
```

### Test Different Time Horizons

```python
# 3-day prediction
r = train_asset(s, 'crypto', horizon=3, threshold=2.0)

# 10-day prediction
r = train_asset(s, 'crypto', horizon=10, threshold=2.0)
```

### Add More Assets

Edit the symbol lists in `main()`:

```python
crypto = ['BTC-USD', 'ETH-USD', ...]  # Add more crypto symbols
stocks = ['AAPL', 'GOOGL', ...]        # Add more stock symbols
```

## Troubleshooting

### Low Accuracy for Specific Assets
Some assets (like MATIC-USD) show lower accuracy due to:
- Higher volatility and noise
- Less predictable price patterns
- Smaller market cap / lower liquidity

### Different Results Than Published
Minor variations (±2-3%) are expected due to:
- Updated market data (new trading days)
- Random seed variations in SMOTE
- Different train/test split boundaries

### Missing Data Errors
Ensure internet connectivity for Yahoo Finance API access.

## Files Reference

| File | Description |
|------|-------------|
| `train_final_publication.py` | Main validation script |
| `train_optimized_final.py` | Alternative with 3% threshold |
| `train_publication_models.py` | Walk-forward validation version |
| `model_results/final_publication_results.json` | Detailed results JSON |
| `model_results/VALIDATION_REPORT.md` | Full validation report |

## Citation

If you use these validation results, please cite:

```
IntelliTradeAI: A Tri-Signal Fusion Framework for Explainable 
AI-Powered Financial Market Prediction
IEEE SoutheastCon 2026
```
