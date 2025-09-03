"""
IntelliTradeAI Dashboard
Streamlit interface for the AI trading agent
"""

# app/dashboard.py
import os, time, json
import streamlit as st
import pandas as pd

from data.data_ingestion import DataIngestion
from models.model_comparison import compare_models
from backtest.backtesting_engine import simulate_long_flat, proba_to_signal

EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR","experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

st.set_page_config(page_title="IntelliTradeAI", layout="wide")
st.title("📈 IntelliTradeAI Dashboard")

st.sidebar.header("Inputs")
crypto = st.sidebar.text_input("Crypto symbols (comma)", "BTC,ETH,FET")
stocks = st.sidebar.text_input("Stock symbols (comma)", "AAPL,MSFT,NVDA")
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d","1h","30m","5m"], index=0)

tabs = st.tabs(["Data","Compare Models","Backtest"])

ing = DataIngestion()

with tabs[0]:
    st.subheader("Fetch data")
    if st.button("Load"):
        symbols_c = [s.strip().upper() for s in crypto.split(",") if s.strip()]
        symbols_s = [s.strip().upper() for s in stocks.split(",") if s.strip()]
        with st.spinner("Pulling…"):
            DATA = ing.fetch_mixed_data(symbols_c, symbols_s, period=period, interval=interval)
        st.success(f"Loaded {len(DATA)} series")
        for s, df in (DATA or {}).items():
            st.markdown(f"**{s}** ({len(df)} rows)")
            st.line_chart(df["close"].tail(200))

with tabs[1]:
    st.subheader("Train & compare (uses first crypto symbol)")
    if st.button("Compare"):
        symbols_c = [s.strip().upper() for s in crypto.split(",") if s.strip()]
        if not symbols_c:
            st.error("Add at least one crypto symbol.")
        else:
            dfmap = ing.fetch_mixed_data(symbols_c[:1], [], period=period, interval=interval)
            sym, df = list(dfmap.items())[0]
            scoreboard, best_model, best_path = compare_models(df)
            st.write("**Scoreboard:**")
            st.dataframe(scoreboard, use_container_width=True)
            st.success(f"Best: {best_model} ({best_path})")
            # save result for thesis
            rid = str(int(time.time()))
            with open(os.path.join(EXPERIMENTS_DIR, f"compare_{rid}.json"), "w") as f:
                json.dump({"symbol":sym,"scoreboard":scoreboard.to_dict("records"),"best":best_model, "path": best_path}, f, indent=2)

with tabs[2]:
    st.subheader("Backtest a probability signal file (or generate quickly)")
    st.caption("For a quick demo, we create a simple thresholded signal from pct_change.")
    thr = st.slider("Threshold (for demo proba→signal)", 0.5, 0.7, 0.55, 0.01)
    if st.button("Run backtest (demo)"):
        dfmap = ing.fetch_mixed_data([ "BTC" ], [], period=period, interval=interval)
        sym, df = list(dfmap.items())[0]
        demo_proba = df["close"].pct_change().rolling(5).mean().fillna(0)
        # normalize to [0,1]
        demo_proba = (demo_proba - demo_proba.min()) / (demo_proba.max()-demo_proba.min() + 1e-9)
        sig = proba_to_signal(demo_proba, threshold=thr)
        metrics, equity, trades = simulate_long_flat(df["close"], sig)
        st.json(metrics)
        st.line_chart(equity.set_index("date")["equity"])
