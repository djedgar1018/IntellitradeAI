# features/feature_engineering.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

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

    # technicals
    x["rsi"] = RSIIndicator(close=x["close"]).rsi()
    x["ema20"] = EMAIndicator(close=x["close"], window=20).ema_indicator()
    macd = MACD(close=x["close"])
    x["macd"] = macd.macd()
    x["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(close=x["close"])
    x["bb_mavg"] = bb.bollinger_mavg()
    x["bb_high"] = bb.bollinger_hband()
    x["bb_low"] = bb.bollinger_lband()

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
