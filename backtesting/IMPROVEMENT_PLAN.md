# Paper Trading Experiment Results & Improvement Plan

## Executive Summary

A 6-month paper trading experiment was conducted from July 31, 2025 to January 27, 2026 with $10,000 allocated to each of three portfolios: Stocks, Crypto, and Forex.

### Overall Results

| Portfolio | Initial | Final | Return | Win Rate | Max DD | Sharpe |
|-----------|---------|-------|--------|----------|--------|--------|
| **Stocks** | $10,000 | $10,071 | +0.71% | 40.3% | 5.15% | 0.23 |
| **Crypto** | $10,000 | $7,773 | -22.27% | 21.1% | 24.24% | -2.08 |
| **Forex** | $10,000 | $7,978 | -20.22% | 44.4% | 20.26% | -1.46 |
| **TOTAL** | $30,000 | $25,822 | -13.93% | 31.2% | - | - |

**Status**: No portfolio hit the 40% max drawdown threshold, but overall performance requires significant improvement.

---

## Critical Issues Identified

### 1. Low Win Rates Across All Portfolios
- **Stocks**: 40.3% (needs 50%+)
- **Crypto**: 21.1% (critically low)
- **Forex**: 44.4% (close to breakeven)

**Root Cause**: Entry signals are too sensitive, triggering trades on minor indicator crossovers without confirming trend direction.

### 2. Poor Risk/Reward Ratios
- **Profit Factor < 1.0** for Crypto and Forex means losses exceed gains
- Average winning trade not large enough to offset losing trades

### 3. Over-Trading
- 202 total trades in 6 months = ~1.5 trades/day
- High frequency leads to accumulated transaction costs and whipsaws

### 4. No Trend Filter
- Strategy trades both directions without confirming overall market trend
- Counter-trend trades have lower probability of success

---

## Improvement Recommendations

### Phase 1: Entry Signal Improvements (Priority: Critical)

#### A. Add Trend Filter
```python
# Only take BUY signals when price > SMA_50 (uptrend)
# Only take SELL signals when price < SMA_50 (downtrend)
if signal == 'BUY' and current_price < sma_50:
    signal = 'HOLD'  # Avoid counter-trend trades
```

**Expected Impact**: +15-20% improvement in win rate

#### B. Increase Confidence Threshold
- Current: 60% confidence required
- Recommended: 75% confidence required for crypto, 70% for stocks
- This will reduce trade frequency but improve quality

#### C. Add Volume Confirmation
```python
# Require above-average volume for valid signals
volume_sma = df['Volume'].rolling(20).mean()
if current_volume < volume_sma * 1.5:
    confidence *= 0.8  # Reduce confidence for low volume signals
```

### Phase 2: Risk Management Improvements (Priority: High)

#### A. Dynamic Stop Loss Based on ATR
```python
# Instead of fixed 5% stop loss
stop_distance = atr * 2.0  # 2x ATR for stop loss
take_profit_distance = atr * 3.0  # 3x ATR for take profit (1.5:1 R:R)
```

**Expected Impact**: Better adaptation to asset volatility

#### B. Position Sizing Based on Volatility
```python
# Reduce position size for volatile assets
base_risk = 0.02  # 2% account risk per trade
volatility_factor = current_atr / average_atr
adjusted_risk = base_risk / volatility_factor
```

#### C. Add Trailing Stops
```python
# Lock in profits as trade moves favorably
if unrealized_pnl > entry_price * 0.03:  # 3% profit
    stop_loss = entry_price  # Move stop to breakeven
if unrealized_pnl > entry_price * 0.05:  # 5% profit
    stop_loss = entry_price * 1.02  # Lock in 2% profit
```

### Phase 3: Asset-Specific Optimizations (Priority: Medium)

#### A. Crypto-Specific Improvements
1. **Add Fear & Greed Index filter**: Only buy when index < 30 (fear)
2. **Add on-chain metrics**: Network activity, whale movements
3. **Reduce trading frequency**: Crypto is 24/7 but signals should be daily only
4. **Increase required RSI extremes**: Buy only when RSI < 25 (vs 30)

#### B. Stock-Specific Improvements
1. **Add sector rotation filter**: Trade with sector momentum
2. **Earnings calendar awareness**: Avoid new positions before earnings
3. **Add relative strength**: Only trade stocks outperforming SPY

#### C. Forex-Specific Improvements
1. **Add session timing**: Trade only during active sessions (London/NY overlap)
2. **Reduce pairs traded**: Focus on major pairs only (EUR/USD, GBP/USD)
3. **Add interest rate differential awareness**

### Phase 4: Portfolio Management (Priority: Medium)

#### A. Correlation Management
```python
# Avoid entering correlated positions simultaneously
# If BTC position open, don't add ETH until BTC closes
correlation_threshold = 0.7
if correlation(new_asset, existing_positions) > correlation_threshold:
    skip_trade = True
```

#### B. Maximum Portfolio Heat
```python
# Limit total portfolio risk exposure
max_open_risk = 0.10  # 10% max portfolio at risk
current_risk = sum(position_risks)
if current_risk + new_position_risk > max_open_risk:
    skip_trade = True
```

---

## Implementation Priority

| Phase | Improvement | Expected Impact | Effort |
|-------|------------|-----------------|--------|
| 1A | Trend Filter | +20% win rate | Low |
| 1B | Higher Confidence | -30% trades, +10% win rate | Low |
| 2A | ATR-based Stops | +15% profit factor | Medium |
| 2C | Trailing Stops | +10% average win | Medium |
| 3A | Crypto Filters | +25% crypto win rate | Medium |
| 4A | Correlation Mgmt | -5% max drawdown | High |

---

## Target Metrics After Improvements

| Metric | Current | Target |
|--------|---------|--------|
| Overall Return | -13.93% | +15-25% |
| Win Rate | 31.2% | 50-55% |
| Profit Factor | 0.39-1.07 | 1.5-2.0 |
| Max Drawdown | 24.24% | <15% |
| Sharpe Ratio | -2.08 to 0.23 | >1.0 |

---

## Next Steps

1. **Immediate**: Implement Phase 1A (Trend Filter) and Phase 1B (Higher Confidence)
2. **Week 1**: Add ATR-based stops and test on paper
3. **Week 2**: Implement trailing stops
4. **Week 3-4**: Add asset-specific optimizations
5. **Month 2**: Full re-backtest with all improvements
6. **Month 3**: Begin live paper trading with improved strategy

---

## Conclusion

The current strategy failed to produce positive returns overall, primarily due to:
1. Trading against the trend
2. Low confidence entry signals generating too many trades
3. Fixed stop losses not adapting to market volatility

With the recommended improvements, particularly the trend filter and dynamic stops, the strategy should achieve profitability. The goal of growing the account will require disciplined implementation of these changes and continuous monitoring.

The 40% max drawdown threshold was NOT breached by any portfolio, indicating the basic risk management framework is sound. The issue is signal quality, not risk control.
