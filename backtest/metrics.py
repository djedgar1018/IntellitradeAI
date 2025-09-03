"""
Backtesting Metrics
Comprehensive metrics calculation and visualization for backtest results
"""

# backtest/metrics.py
import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())

def sharpe_ratio(returns: pd.Series, rf_daily=0.0):
    if returns.std(ddof=1) == 0: return 0.0
    return float((returns.mean() - rf_daily) / returns.std(ddof=1) * np.sqrt(252))

def cagr(equity: pd.Series, periods_per_year=252):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    return float((1 + total_ret) ** (1/years) - 1) if years > 0 else 0.0

def win_rate(trade_returns: pd.Series):
    if len(trade_returns)==0: return 0.0
    wins = (trade_returns > 0).sum()
    return float(wins / len(trade_returns))
