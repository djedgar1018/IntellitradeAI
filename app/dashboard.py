# --- make project root importable when running from app/ ---
import os, sys, time, json, glob
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.data_ingestion import DataIngestion
from models.model_comparison import compare_models
from backtest.features.feature_engineering import build_features
from backtest.backtesting_engine import simulate_long_flat, proba_to_signal

EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def latest_compare_record():
    """Return the last compare_*.json dict or None."""
    paths = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "compare_*.json")))
    if not paths:
        return None
    with open(paths[-1], "r") as f:
        return json.load(f)

def save_compare_record(symbol, scoreboard_df, best_model, best_path):
    ts = int(time.time())
    rec = {
        "ts": ts,
        "symbol": symbol,
        "best_model": best_model,
        "best_model_path": best_path,
        "scoreboard": scoreboard_df.to_dict("records"),
    }
    out = os.path.join(EXPERIMENTS_DIR, f"compare_{ts}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    return out

st.set_page_config(page_title="IntelliTradeAI", layout="wide")
st.title("📈 IntelliTradeAI Dashboard")

# --- Sidebar controls ---
st.sidebar.header("Inputs")
crypto_text = st.sidebar.text_input("Crypto symbols (comma)", "BTC,ETH,FET")
stock_text  = st.sidebar.text_input("Stock symbols (comma)",  "AAPL,MSFT,NVDA")
period   = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d","1h","30m","5m"], index=0)

# Parse user lists
crypto_syms = [s.strip().upper() for s in crypto_text.split(",") if s.strip()]
stock_syms  = [s.strip().upper() for s in stock_text.split(",") if s.strip()]

ing = DataIngestion()

tab_data, tab_compare, tab_backtest = st.tabs(["Data (CMC + Yahoo)","Compare Models","Backtest Best Model"])

# =========================
# 1) DATA TAB
# =========================
with tab_data:
    st.subheader("Fetch Data from both sources")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Mixed (Crypto via CMC + Stocks via Yahoo)"):
            with st.spinner("Fetching…"):
                MIX = ing.fetch_mixed_data(
                    crypto_symbols=crypto_syms,
                    stock_symbols=stock_syms,
                    period=period, interval=interval
                )
            st.success(f"Loaded {len(MIX)} series")
            for sym, df in MIX.items():
                st.markdown(f"**{sym}** ({len(df)} rows)")
                # Create proper price chart with axis labels
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index.tail(250),
                    y=df["close"].tail(250),
                    mode='lines',
                    name=f'{sym} Price',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title=f'{sym} Price Chart',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if st.button("Quick-Load BTC from CoinMarketCap"):
            with st.spinner("Pulling BTC OHLCV from CMC…"):
                BTC = ing.fetch_crypto_data(["BTC"], period=period, interval=interval)
            if not BTC:
                st.error("No BTC data returned. Check your CMC_API_KEY in .env/Secrets.")
            else:
                df = BTC["BTC"]
                st.success(f"BTC rows: {len(df)} (source: CMC)")
                # Create proper BTC price chart with axis labels
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index.tail(250),
                    y=df["close"].tail(250),
                    mode='lines',
                    name='BTC Price',
                    line=dict(color='orange')
                ))
                fig.update_layout(
                    title='Bitcoin (BTC) Price Chart',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: Crypto uses CoinMarketCap (CMC_API_KEY required). Stocks use Yahoo Finance via `yfinance`.")

# =========================
# 2) COMPARE MODELS TAB
# =========================
with tab_compare:
    st.subheader("Train & Compare (RF / XGB / LSTM when available)")
    st.write("Select one symbol to compare on (stock preferred if provided).")

    # Choose data source automatically: stock preferred
    candidate = (stock_syms[:1] or crypto_syms[:1])
    if not candidate:
        st.info("Add at least one symbol in the sidebar to enable comparison.")
    else:
        if st.button("Run Comparison"):
            with st.spinner("Loading series & training models…"):
                if stock_syms:
                    data_map = ing.fetch_mixed_data(crypto_symbols=[], stock_symbols=stock_syms[:1], period=period, interval=interval)
                else:
                    data_map = ing.fetch_mixed_data(crypto_symbols=crypto_syms[:1], stock_symbols=[], period=period, interval=interval)

                sym, df = list(data_map.items())[0]
                scoreboard, best_model, best_path = compare_models(df)

            st.success(f"Best model: **{best_model}**")
            st.dataframe(scoreboard, use_container_width=True)

            out = save_compare_record(sym, scoreboard, best_model, best_path)
            st.caption(f"Saved comparison record → `{out}`")

# =========================
# 3) BACKTEST TAB
# =========================
with tab_backtest:
    st.subheader("Backtest using the most recent best model (auto loaded)")
    thr = st.slider("Probability threshold (long when ≥ threshold)", 0.50, 0.70, 0.55, 0.01)

    if st.button("Run Model-Driven Backtest"):
        # 1) Load most recent comparison artifact
        rec = latest_compare_record()
        if not rec:
            st.warning("No compare record found. Run 'Compare Models' first.")
        else:
            model_path = rec.get("best_model_path")
            if not model_path or not os.path.exists(model_path):
                # fallback to cache paths
                for p in ("models/cache/xgb.pkl", "models/cache/rf.pkl"):
                    if os.path.exists(p):
                        model_path = p; break

        if not rec or not model_path or not os.path.exists(model_path):
            st.error("No saved model artifact available.")
        else:
            # 2) Pull a fresh series for the same symbol used in compare
            symbol_to_use = rec.get("symbol", (stock_syms[:1] or crypto_syms[:1] or ["AAPL"]))[0]
            if symbol_to_use in stock_syms:
                data_map = ing.fetch_mixed_data(crypto_symbols=[], stock_symbols=[symbol_to_use], period=period, interval=interval)
            else:
                data_map = ing.fetch_mixed_data(crypto_symbols=[symbol_to_use], stock_symbols=[], period=period, interval=interval)

            sym, df = list(data_map.items())[0]

            # 3) Rebuild features in the same way as training
            X, y, feats, processed = build_features(df, horizon=1)

            try:
                model = joblib.load(model_path)
            except Exception as e:
                st.error(f"Failed loading model: {e}")
                model = None

            if model is None:
                st.stop()

            # 4) Probabilities → signals
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                z = model.decision_function(X)
                proba = 1 / (1 + np.exp(-z))
            else:
                proba = model.predict(X).astype(float)

            proba_s = pd.Series(proba, index=processed.index[:len(proba)])
            signals = (proba_s >= thr).astype(int)

            # 5) Backtest
            metrics, equity, trades = simulate_long_flat(processed["close"], signals)
            st.json(metrics)
            
            # Create proper equity curve chart with axis labels
            equity_df = equity.set_index("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df["equity"],
                mode='lines',
                name='Portfolio Equity',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title='Backtest Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (USD)',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Backtested: {sym} | Model: {os.path.basename(model_path)}")