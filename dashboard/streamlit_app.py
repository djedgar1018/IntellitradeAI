"""
Streamlit Dashboard for AI Trading Agent
Main user interface for the trading agent application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_fetcher import DataFetcher
from utils.indicators import TechnicalIndicators
from utils.data_cleaner import DataCleaner
from utils.explainability import ExplainabilityEngine
# LSTM model temporarily disabled due to TensorFlow compatibility issues
LSTM_AVAILABLE = False
# Temporarily comment out complex model imports to fix import errors
# from models.random_forest_model import make_model as make_rf_model
# from models.xgboost_model import make_model as make_xgb_model
# from models.model_comparison import compare_models
# from backtest.backtesting_engine import simulate_long_flat, proba_to_signal
# from backtest.metrics import BacktestMetrics
from ranking.tool_ranking import ToolRanking
from analysis.cross_market import CrossMarketAnalysis
from config import config

# Page configuration
st.set_page_config(
    page_title="AI Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'explainability_engine' not in st.session_state:
    st.session_state.explainability_engine = ExplainabilityEngine()
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def load_data():
    """Load data from APIs or cache"""
    try:
        with st.spinner("Loading market data..."):
            # Get both crypto and stock data
            data = st.session_state.data_fetcher.get_data(data_type='mixed', use_cache=True)
            
            if not data:
                st.error("No data available. Please check your API keys and internet connection.")
                return None
            
            # Clean and process data
            processed_data = {}
            for symbol, df in data.items():
                if df is not None and not df.empty:
                    # Clean the data
                    cleaned_df = DataCleaner.clean_ohlcv_data(df)
                    
                    # Calculate technical indicators
                    if len(cleaned_df) > 50:  # Ensure enough data for indicators
                        processed_data[symbol] = TechnicalIndicators.calculate_all_indicators(cleaned_df)
                    else:
                        processed_data[symbol] = cleaned_df
            
            st.session_state.last_update = datetime.now()
            return processed_data
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_price_chart(data, symbol):
    """Create price chart with technical indicators"""
    try:
        if symbol not in data or data[symbol].empty:
            return None
        
        df = data[symbol]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Technical Indicators', 'Volume'),
            row_heights=[0.6, 0.25, 0.15]
        )
        
        # Price chart with Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f'{symbol} Price'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if available
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='red', width=1),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Add moving averages
        if 'ema_12' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ema_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        if 'ema_26' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ema_26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            if 'macd_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI/MACD", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return None

def train_models(data, symbol):
    """Train ML models on selected data"""
    try:
        if symbol not in data or len(data[symbol]) < 100:
            st.error(f"Not enough data to train models for {symbol}")
            return False
        
        df = data[symbol]
        
        with st.spinner(f"Training models for {symbol}..."):
            # Initialize models
            rf_model = RandomForestPredictor(model_type='classifier')
            xgb_model = XGBoostPredictor(model_type='classifier')
            
            # Train LSTM (if available)
            if LSTM_AVAILABLE:
                lstm_model = LSTMPredictor()
                try:
                    lstm_model.train(df)
                    st.session_state.models[f'{symbol}_lstm'] = lstm_model
                    st.success("✅ LSTM model trained successfully")
                except Exception as e:
                    st.error(f"❌ LSTM training failed: {str(e)}")
            else:
                st.warning("⚠️ LSTM model not available (TensorFlow not installed)")
            
            # Train Random Forest
            try:
                rf_metrics = rf_model.train(df)
                st.session_state.models[f'{symbol}_rf'] = rf_model
                st.success(f"✅ Random Forest trained successfully (Accuracy: {rf_metrics.get('accuracy', 0):.3f})")
            except Exception as e:
                st.error(f"❌ Random Forest training failed: {str(e)}")
            
            # Train XGBoost
            try:
                xgb_metrics = xgb_model.train(df)
                st.session_state.models[f'{symbol}_xgb'] = xgb_model
                st.success(f"✅ XGBoost trained successfully (Accuracy: {xgb_metrics.get('accuracy', 0):.3f})")
            except Exception as e:
                st.error(f"❌ XGBoost training failed: {str(e)}")
        
        return True
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return False

def get_model_predictions(data, symbol):
    """Get predictions from all trained models"""
    predictions = {}
    
    # Check available models for this symbol
    for model_key, model in st.session_state.models.items():
        if model_key.startswith(symbol) and model.is_trained:
            try:
                signal, confidence = model.get_signal(data[symbol])
                model_name = model_key.split('_')[-1].upper()
                predictions[model_name] = {
                    'signal': signal,
                    'confidence': confidence
                }
            except Exception as e:
                st.error(f"Error getting prediction from {model_key}: {str(e)}")
    
    return predictions

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🤖 AI-Powered Trading Agent")
    st.markdown("*Cross-market analysis with explainable AI and comprehensive backtesting*")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Model Training", "Model Comparison", "Backtesting", "Tool Ranking", "Cross-Market Analysis", "Settings"]
    )
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Cannot proceed without data. Please check your configuration.")
        return
    
    # Display last update time
    if st.session_state.last_update:
        st.sidebar.success(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Page routing
    if page == "Dashboard":
        dashboard_page(data)
    elif page == "Model Training":
        model_training_page(data)
    elif page == "Model Comparison":
        model_comparison_page(data)
    elif page == "Backtesting":
        backtesting_page(data)
    elif page == "Tool Ranking":
        tool_ranking_page()
    elif page == "Cross-Market Analysis":
        cross_market_analysis_page(data)
    elif page == "Settings":
        settings_page()

def dashboard_page(data):
    """Main dashboard page"""
    st.header("📊 Market Dashboard")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crypto_count = sum(1 for symbol in data.keys() if symbol in config.DATA_CONFIG["crypto_symbols"])
        st.metric("Crypto Assets", crypto_count)
    
    with col2:
        stock_count = sum(1 for symbol in data.keys() if symbol in config.DATA_CONFIG["stock_symbols"])
        st.metric("Stock Assets", stock_count)
    
    with col3:
        trained_models = len([k for k in st.session_state.models.keys() if st.session_state.models[k].is_trained])
        st.metric("Trained Models", trained_models)
    
    with col4:
        total_signals = 0
        for symbol in data.keys():
            predictions = get_model_predictions(data, symbol)
            total_signals += len(predictions)
        st.metric("Active Signals", total_signals)
    
    # Symbol selection
    st.subheader("Asset Analysis")
    available_symbols = list(data.keys())
    selected_symbol = st.selectbox("Select Asset", available_symbols)
    
    if selected_symbol and selected_symbol in data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            fig = create_price_chart(data, selected_symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Technical indicators summary
            st.subheader("Technical Analysis")
            
            if len(data[selected_symbol]) > 0:
                indicators_summary = TechnicalIndicators.get_indicator_summary(data[selected_symbol])
                
                for indicator, info in indicators_summary.items():
                    st.write(f"**{indicator}**")
                    if 'signal' in info:
                        signal_color = {
                            'Bullish': 'green',
                            'Bearish': 'red',
                            'Overbought': 'red',
                            'Oversold': 'green',
                            'Neutral': 'gray',
                            'Normal': 'blue',
                            'High': 'orange',
                            'Low': 'orange'
                        }.get(info['signal'], 'black')
                        
                        st.markdown(f"<span style='color:{signal_color}'>{info['signal']}</span>", 
                                  unsafe_allow_html=True)
                    
                    if 'value' in info:
                        st.write(f"Value: {info['value']:.2f}")
                    
                    st.write("---")
            
            # Model predictions
            st.subheader("Model Predictions")
            predictions = get_model_predictions(data, selected_symbol)
            
            if predictions:
                for model_name, pred_info in predictions.items():
                    signal = pred_info['signal']
                    confidence = pred_info['confidence']
                    
                    signal_color = {
                        'buy': 'green',
                        'sell': 'red',
                        'hold': 'gray'
                    }.get(signal, 'black')
                    
                    st.write(f"**{model_name}**")
                    st.markdown(f"<span style='color:{signal_color}'>{signal.upper()}</span>", 
                              unsafe_allow_html=True)
                    st.write(f"Confidence: {confidence:.3f}")
                    st.write("---")
            else:
                st.info("No trained models available for this asset. Go to Model Training to train models.")
    
    # Market alerts
    st.subheader("🚨 Market Alerts")
    
    alerts = []
    for symbol, df in data.items():
        if len(df) > 0:
            latest = df.iloc[-1]
            
            # RSI alerts
            if 'rsi' in df.columns and not pd.isna(latest['rsi']):
                if latest['rsi'] > 70:
                    alerts.append(f"⚠️ {symbol}: RSI overbought ({latest['rsi']:.1f})")
                elif latest['rsi'] < 30:
                    alerts.append(f"📈 {symbol}: RSI oversold ({latest['rsi']:.1f})")
            
            # Volume alerts
            if 'volume_ratio' in df.columns and not pd.isna(latest['volume_ratio']):
                if latest['volume_ratio'] > 2.0:
                    alerts.append(f"📊 {symbol}: High volume activity ({latest['volume_ratio']:.1f}x)")
    
    if alerts:
        for alert in alerts[:10]:  # Show top 10 alerts
            st.write(alert)
    else:
        st.info("No market alerts at this time.")

def model_training_page(data):
    """Model training page"""
    st.header("🤖 Model Training")
    
    # Symbol selection
    available_symbols = list(data.keys())
    selected_symbol = st.selectbox("Select Asset for Training", available_symbols)
    
    if selected_symbol and selected_symbol in data:
        # Data quality check
        quality_metrics = DataCleaner.validate_data_quality(data[selected_symbol])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Quality Score", f"{quality_metrics['quality_score']:.2f}")
        with col2:
            st.metric("Total Rows", quality_metrics['total_rows'])
        with col3:
            st.metric("Data Issues", len(quality_metrics['issues']))
        
        # Show data quality issues
        if quality_metrics['issues']:
            st.warning("Data Quality Issues:")
            for issue in quality_metrics['issues']:
                st.write(f"• {issue}")
        
        # Training options
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            if LSTM_AVAILABLE:
                train_lstm = st.checkbox("Train LSTM Model", value=True)
            else:
                st.checkbox("Train LSTM Model", value=False, disabled=True, help="TensorFlow not available")
            train_rf = st.checkbox("Train Random Forest", value=True)
            train_xgb = st.checkbox("Train XGBoost", value=True)
        
        with col2:
            lookahead = st.slider("Prediction Lookahead (days)", 1, 10, 1)
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            if train_models(data, selected_symbol):
                st.success("Model training completed!")
                st.balloons()
        
        # Show existing models
        st.subheader("Trained Models")
        
        model_status = []
        for model_key, model in st.session_state.models.items():
            if model_key.startswith(selected_symbol):
                model_name = model_key.split('_')[-1].upper()
                status = "✅ Trained" if model.is_trained else "❌ Not Trained"
                model_status.append({"Model": model_name, "Status": status})
        
        if model_status:
            st.dataframe(pd.DataFrame(model_status), use_container_width=True)
        else:
            st.info("No models trained for this asset yet.")

def model_comparison_page(data):
    """Model comparison page"""
    st.header("⚖️ Model Comparison")
    
    # Symbol selection
    available_symbols = list(data.keys())
    selected_symbol = st.selectbox("Select Asset", available_symbols)
    
    if selected_symbol and selected_symbol in data:
        # Get trained models for this symbol
        symbol_models = {k: v for k, v in st.session_state.models.items() 
                        if k.startswith(selected_symbol) and v.is_trained}
        
        if symbol_models:
            # Create model comparison
            comparison = ModelComparison()
            
            # Add models to comparison
            for model_key, model in symbol_models.items():
                model_name = model_key.split('_')[-1].upper()
                comparison.add_model(model_name, model)
            
            # Run comparison
            comparison_results = comparison.compare_models(data[selected_symbol])
            
            # Display comparison dashboard
            comparison.display_comparison_dashboard(data[selected_symbol], comparison_results)
            
            # Explainability section
            st.subheader("🔍 Model Explainability")
            
            # Select model for explanation
            model_names = list(comparison_results.keys())
            selected_model = st.selectbox("Select Model for Explanation", model_names)
            
            if selected_model and comparison_results[selected_model]['status'] == 'success':
                # Create explanation (simplified for demo)
                explanation = {
                    'prediction': comparison_results[selected_model]['signal'],
                    'confidence': comparison_results[selected_model]['confidence'],
                    'expected_value': 0.5,
                    'shap_values': np.random.randn(10),  # Placeholder
                    'feature_names': ['RSI', 'MACD', 'Volume', 'Price_Change', 'Volatility', 
                                    'EMA_12', 'EMA_26', 'BB_Position', 'Momentum', 'ATR'],
                    'feature_values': np.random.randn(10)  # Placeholder
                }
                
                # Display explainability dashboard
                st.session_state.explainability_engine.create_explainability_dashboard(
                    selected_model, explanation, selected_symbol
                )
        else:
            st.info("No trained models available for this asset. Please train models first.")

def backtesting_page(data):
    """Backtesting page"""
    st.header("📈 Backtesting Engine")
    
    # Symbol selection
    available_symbols = list(data.keys())
    selected_symbol = st.selectbox("Select Asset", available_symbols)
    
    if selected_symbol and selected_symbol in data:
        # Get trained models
        symbol_models = {k: v for k, v in st.session_state.models.items() 
                        if k.startswith(selected_symbol) and v.is_trained}
        
        if symbol_models:
            # Backtesting parameters
            st.subheader("Backtesting Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                initial_capital = st.number_input("Initial Capital", value=10000, min_value=1000)
                commission = st.number_input("Commission (%)", value=0.1, min_value=0.0, max_value=5.0) / 100
                
            with col2:
                start_date = st.date_input("Start Date", 
                                         value=datetime.now() - timedelta(days=180))
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Model selection
            model_names = [k.split('_')[-1].upper() for k in symbol_models.keys()]
            selected_model = st.selectbox("Select Model", model_names)
            
            # Run backtest
            if st.button("🚀 Run Backtest", type="primary"):
                try:
                    # Initialize backtesting engine
                    backtest_engine = BacktestingEngine(
                        initial_capital=initial_capital,
                        commission=commission
                    )
                    
                    # Get model
                    model_key = f"{selected_symbol}_{selected_model.lower()}"
                    model = st.session_state.models[model_key]
                    
                    # Filter data by date range
                    test_data = data[selected_symbol]
                    test_data = test_data[
                        (test_data.index >= pd.to_datetime(start_date)) & 
                        (test_data.index <= pd.to_datetime(end_date))
                    ]
                    
                    if len(test_data) < 30:
                        st.error("Not enough data for backtesting. Please select a longer time range.")
                        return
                    
                    with st.spinner("Running backtest..."):
                        # Run backtest
                        results = backtest_engine.run_backtest(model, test_data, selected_symbol)
                        
                        if results:
                            # Calculate metrics
                            metrics = BacktestMetrics.calculate_metrics(results)
                            
                            # Display results
                            st.subheader("📊 Backtest Results")
                            
                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", f"{metrics['total_return']:.2%}")
                            with col2:
                                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                            with col3:
                                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                            with col4:
                                st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                            
                            # Performance chart
                            BacktestMetrics.plot_performance(results)
                            
                            # Trade analysis
                            st.subheader("Trade Analysis")
                            trades_df = BacktestMetrics.get_trades_summary(results)
                            st.dataframe(trades_df, use_container_width=True)
                            
                        else:
                            st.error("Backtest failed. Please check your model and data.")
                            
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
        else:
            st.info("No trained models available for backtesting. Please train models first.")

def tool_ranking_page():
    """Tool ranking page"""
    st.header("🏆 Trading Tool Ranking")
    
    # Initialize tool ranking
    tool_ranking = ToolRanking()
    
    # Get ranking results
    ranking_results = tool_ranking.get_comprehensive_ranking()
    
    # Display ranking dashboard
    tool_ranking.display_ranking_dashboard(ranking_results)

def cross_market_analysis_page(data):
    """Cross-market analysis page"""
    st.header("🌐 Cross-Market Analysis")
    
    # Initialize cross-market analysis
    cross_market = CrossMarketAnalysis()
    
    # Run cross-market analysis
    with st.spinner("Analyzing cross-market correlations..."):
        analysis_results = cross_market.run_cross_market_analysis(data)
    
    # Display analysis dashboard
    cross_market.display_analysis_dashboard(analysis_results)

def settings_page():
    """Settings page"""
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    # Display current API key status (masked)
    coinmarketcap_key = config.COINMARKETCAP_API_KEY
    if coinmarketcap_key != "default_key":
        st.success("✅ CoinMarketCap API Key: Configured")
    else:
        st.warning("⚠️ CoinMarketCap API Key: Not configured")
    
    st.info("API keys are managed through environment variables. Please set COINMARKETCAP_API_KEY in your .env file.")
    
    # Trading Parameters
    st.subheader("Trading Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Buy Threshold", value=config.TRADING_CONFIG["buy_threshold"], 
                       min_value=0.0, max_value=1.0, step=0.1)
        st.number_input("Sell Threshold", value=config.TRADING_CONFIG["sell_threshold"], 
                       min_value=0.0, max_value=1.0, step=0.1)
    
    with col2:
        st.number_input("Risk Tolerance", value=config.TRADING_CONFIG["risk_tolerance"], 
                       min_value=0.01, max_value=0.1, step=0.01)
        st.number_input("Stop Loss", value=config.TRADING_CONFIG["stop_loss"], 
                       min_value=0.01, max_value=0.2, step=0.01)
    
    # Data Configuration
    st.subheader("Data Configuration")
    
    # Crypto symbols
    crypto_symbols = st.multiselect(
        "Crypto Symbols", 
        options=["BTC", "ETH", "ADA", "SOL", "MATIC", "DOT", "LINK", "UNI"],
        default=config.DATA_CONFIG["crypto_symbols"]
    )
    
    # Stock symbols
    stock_symbols = st.multiselect(
        "Stock Symbols", 
        options=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
        default=config.DATA_CONFIG["stock_symbols"]
    )
    
    # Cache settings
    st.subheader("Cache Settings")
    cache_duration = st.slider("Cache Duration (minutes)", 1, 60, 
                              config.DATA_CONFIG["cache_duration"] // 60)
    
    # System Information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Data Sources:** Yahoo Finance, CoinMarketCap")
        models_text = "Random Forest, XGBoost"
        if LSTM_AVAILABLE:
            models_text = "LSTM, " + models_text
        st.info(f"**ML Models:** {models_text}")
        st.info("**Technical Indicators:** RSI, MACD, Bollinger Bands, EMA")
    
    with col2:
        st.info("**Backtesting:** Custom engine with comprehensive metrics")
        st.info("**Explainability:** SHAP-based model interpretation")
        st.info("**Cross-Market:** Correlation and arbitrage analysis")
    
    # Clear cache button
    if st.button("🗑️ Clear Cache", type="secondary"):
        if os.path.exists(config.PATHS["crypto_data"]):
            os.remove(config.PATHS["crypto_data"])
        if os.path.exists(config.PATHS["stock_data"]):
            os.remove(config.PATHS["stock_data"])
        st.success("Cache cleared successfully!")

if __name__ == "__main__":
    main()
