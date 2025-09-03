"""
Model Comparison Module
Provides functionality to compare different ML models side by side
"""
# models/model_comparison.py
import os, joblib, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from backtest.features.feature_engineering import build_features
from .random_forest_model import make_model as rf_model
from .xgboost_model import make_model as xgb_model
from .lstm_model import make_model as lstm_model, to_sequences, SkipLSTM

SAVE_DIR = "models/cache"
os.makedirs(SAVE_DIR, exist_ok=True)

def _evaluate_binary(y_true, y_pred, y_proba=None):
    out = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = None
    return out

def compare_models(df, horizon=1, window=20):
    """
    Trains RF, XGB, and (optionally) LSTM on the same engineered features with a
    time-series split; returns a scoreboard DataFrame and best model path.
    """
    X, y, feats, processed = build_features(df, horizon=horizon)
    tscv = TimeSeriesSplit(n_splits=3)

    rows, artifacts = [], {}

    # Random Forest
    rf = rf_model()
    tr_idx, te_idx = list(tscv.split(X))[-1]
    rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    p = rf.predict(X.iloc[te_idx])
    proba = getattr(rf, "predict_proba", lambda v: None)(X.iloc[te_idx])
    proba = proba[:,1] if proba is not None else None
    m = _evaluate_binary(y.iloc[te_idx], p, proba)
    rf_path = os.path.join(SAVE_DIR, "rf.pkl"); joblib.dump(rf, rf_path)
    rows.append({"model":"RandomForest", **m}); artifacts["RandomForest"] = rf_path

    # XGBoost
    try:
        xgb = xgb_model()
        xgb.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = xgb.predict(X.iloc[te_idx])
        proba = getattr(xgb, "predict_proba", lambda v: None)(X.iloc[te_idx])
        proba = proba[:,1] if proba is not None else None
        m = _evaluate_binary(y.iloc[te_idx], p, proba)
        xgb_path = os.path.join(SAVE_DIR, "xgb.pkl"); joblib.dump(xgb, xgb_path)
        rows.append({"model":"XGBoost", **m}); artifacts["XGBoost"] = xgb_path
    except RuntimeError:
        rows.append({"model":"XGBoost", "accuracy":None,"precision":None,"recall":None,"f1":None,"roc_auc":None})

    # LSTM (optional)
    try:
        model = lstm_model(input_dim=X.shape[1])
        Xs, ys = to_sequences(X, y, window=window)
        split = int(len(Xs)*0.8)
        model.fit(Xs[:split], ys[:split], epochs=5, batch_size=64, verbose=0)
        pred = (model.predict(Xs[split:], verbose=0) > 0.5).astype(int).ravel()
        m = _evaluate_binary(ys[split:], pred, None)
        lstm_path = os.path.join(SAVE_DIR, "lstm.h5"); model.save(lstm_path)
        rows.append({"model":"LSTM", **m}); artifacts["LSTM"] = lstm_path
    except SkipLSTM:
        rows.append({"model":"LSTM", "accuracy":None,"precision":None,"recall":None,"f1":None,"roc_auc":None})

    scoreboard = pd.DataFrame(rows).sort_values(by=["accuracy","f1"], ascending=False, na_position="last")
    best_row = scoreboard.iloc[0]
    best_model = best_row["model"]
    best_path = artifacts.get(best_model)
    return scoreboard, best_model, best_path

