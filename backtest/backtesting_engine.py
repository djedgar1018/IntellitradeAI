"""
Backtesting Engine
Custom backtesting engine with user-defined metrics and comprehensive analysis
"""

# backtest/backtesting_engine.py
import pandas as pd
from .metrics import max_drawdown, sharpe_ratio, cagr, win_rate

def simulate_long_flat(prices: pd.Series, signals: pd.Series, start_capital=10_000, fee_bps=5):
    """signals: 1 = long, 0 = flat"""
    prices, signals = prices.align(signals, join="inner")
    cash, position = float(start_capital), 0.0
    equity, trades = [], []

    for t in range(len(prices)):
        price = float(prices.iloc[t])
        sig   = int(signals.iloc[t])

        # enter
        if position == 0.0 and sig == 1:
            qty = cash / price
            fee = cash * (fee_bps/10000)
            cash = -fee
            position = qty
            trades.append({"t": prices.index[t], "side":"buy", "price":price})

        # exit
        if position > 0.0 and sig == 0:
            proceeds = position * price
            fee = proceeds * (fee_bps/10000)
            cash = proceeds - fee
            trades.append({"t": prices.index[t], "side":"sell", "price":price})
            position = 0.0

        equity.append(cash + position*price)

    # liquidate
    if position > 0.0:
        price = float(prices.iloc[-1])
        proceeds = position * price
        fee = proceeds * (fee_bps/10000)
        cash = proceeds - fee
        position = 0.0
        equities = pd.Series(equity + [cash], index=list(prices.index)+[prices.index[-1]])
    else:
        equities = pd.Series(equity, index=prices.index)

    # metrics
    ret = pd.Series(equities).pct_change().dropna()
    metrics = {
        "final_equity": float(equities.iloc[-1]),
        "total_return_pct": float((equities.iloc[-1]/equities.iloc[0]-1)*100),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(equities),
        "cagr": cagr(equities),
        "n_trades": len(trades),
    }
    return metrics, pd.DataFrame({"date": equities.index, "equity": equities.values}), pd.DataFrame(trades)

def proba_to_signal(proba: pd.Series, threshold=0.55):
    return (proba >= threshold).astype(int)
