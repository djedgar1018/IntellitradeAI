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
            if not dfmap:
                st.error("No data available. Please check symbols and try again.")
            else:
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
    st.subheader("Backtest using best model probabilities")
    thr = st.slider("Prob. threshold (long when ≥ threshold)", 0.50, 0.70, 0.55, 0.01)

    if st.button("Run backtest (best model)"):
        # 1) pick one symbol (stock or crypto)
        symbols_s = [s.strip().upper() for s in stocks.split(",") if s.strip()]
        symbols_c = [s.strip().upper() for s in crypto.split(",") if s.strip()]
        # prefer stock if provided, else crypto
        if symbols_s:
            data_map = ing.fetch_mixed_data(crypto_symbols=[], stock_symbols=symbols_s[:1], period=period, interval=interval)
        else:
            data_map = ing.fetch_mixed_data(crypto_symbols=symbols_c[:1], stock_symbols=[], period=period, interval=interval)

        if not data_map:
            st.error("No data available. Please check symbols and try again.")
        else:
            sym, df = list(data_map.items())[0]

            # 2) Rebuild features to match training
            X, y, feats, processed = build_features(df, horizon=1)

            # 3) Load best model artifact from the last compare run
            #    (If you want it automatic, read the most recent experiments/compare_*.json)
            #    For now, try RF by default and fall back gracefully:
            model_path_candidates = [
                "models/cache/xgb.pkl",
                "models/cache/rf.pkl",
            ]
            model = None
            for p in model_path_candidates:
                if os.path.exists(p):
                    model = joblib.load(p)
                    break
            if model is None:
                st.error("No saved model found. Run Compare Models first.")
            else:
                # 4) Generate probabilities
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[:, 1]
                elif hasattr(model, "decision_function"):
                    z = model.decision_function(X)
                    proba = 1 / (1 + np.exp(-z))
                else:
                    # fallback to binary preds
                    proba = model.predict(X).astype(float)

                proba_s = pd.Series(proba, index=processed.index[:len(proba)])
                signals = (proba_s >= thr).astype(int)

                # 5) Backtest
                metrics, equity, trades = simulate_long_flat(processed["close"], signals)
                st.json(metrics)
                st.line_chart(equity.set_index("date")["equity"])
