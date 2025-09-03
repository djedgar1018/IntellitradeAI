# features/feature_engineering.py
import pandas as pd
import numpy as np
# Temporarily disable ta imports due to missing library
# from ta.momentum import RSIIndicator
# from ta.trend import MACD, EMAIndicator
# from ta.volatility import BollingerBands

def build_features(df: pd.DataFrame, horizon: int = 1):
    """
    Input df with columns: open, high, low, close, volume (index = datetime)
    Returns: X, y, feats, processed_df
    """
    x = df.copy().sort_index()

    # basic returns & lags
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_5"] = x["close"].pct_change(5)
    x["vol_chg"] = x["volume"].pct_change()

    for lag in (1,2,3,5):
        x[f"close_lag{lag}"] = x["close"].shift(lag)

    # price action
    x["body"] = x["close"] - x["open"]
    x["range"] = x["high"] - x["low"]
    x["upper_shadow"] = x["high"] - x[["close","open"]].max(axis=1)
    x["lower_shadow"] = x[["close","open"]].min(axis=1) - x["low"]

    # technicals (simplified without ta library)
    # Simple RSI approximation
    delta = x["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    x["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # Simple EMA
    x["ema20"] = x["close"].ewm(span=20).mean()
    
    # Simple MACD approximation
    ema12 = x["close"].ewm(span=12).mean()
    ema26 = x["close"].ewm(span=26).mean()
    x["macd"] = ema12 - ema26
    x["macd_diff"] = x["macd"].ewm(span=9).mean()
    
    # Simple Bollinger Bands
    x["bb_mavg"] = x["close"].rolling(window=20).mean()
    bb_std = x["close"].rolling(window=20).std()
    x["bb_high"] = x["bb_mavg"] + (bb_std * 2)
    x["bb_low"] = x["bb_mavg"] - (bb_std * 2)

    # time/cycle
    x["dow"] = x.index.dayofweek
    if hasattr(x.index, "hour"):
        x["hour"] = getattr(x.index, "hour", pd.Series(0, index=x.index))

    # target: next-step up/down
    x["target"] = (x["close"].shift(-horizon) > x["close"]).astype(int)

    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    feats = [c for c in x.columns if c not in ("target")]
    X = x[feats].copy()
    y = x["target"].copy()
    return X, y, feats, x
