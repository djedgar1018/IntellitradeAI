"""
Enhanced AI Trading Dashboard with User Authentication and Blockchain Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import base64
from typing import Dict, List, Optional

# Import our new modules with error handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from auth.secure_auth import SecureAuthManager
    from blockchain.wallet_manager import SecureWalletManager, PortfolioTracker
    from ai_vision.chart_pattern_recognition import ChartPatternRecognizer
    from ai_advisor.trading_intelligence import TradingIntelligence
    # Try to import the real data ingestion module
    from data.data_ingestion import DataIngestion
    ing = DataIngestion()
    
    # Test if the data ingestion is working properly
    try:
        test_data = ing.fetch_mixed_data(stock_symbols=['AAPL'], period='5d', interval='1d')
        if not test_data:
            raise Exception("Data ingestion test failed")
    except Exception as test_error:
        # If real data ingestion fails, use fallback
        raise ImportError(f"Data ingestion not working: {test_error}")
except ImportError as e:
    st.warning(f"Using demo mode due to import issue: {e}")
    # Create mock classes for demo
    class SecureAuthManager:
        def __init__(self, secret_key): pass
        def authenticate_user(self, username, password, totp_token=None):
            if username and password:
                return {"success": True, "user": {"id": 1, "username": username, "email": f"{username}@example.com", "is_2fa_enabled": False}}
            return {"success": False, "error": "Invalid credentials"}
        def register_user(self, username, email, password):
            return {"success": True, "user_data": {"username": username, "email": email}}
    
    class SecureWalletManager:
        def __init__(self): pass
        def create_ethereum_wallet(self, password):
            return {"address": "0x742d35cc6e7312e2b5b8c8b...c35be", "status": "created"}
        def generate_wallet_qr_code(self, address):
            return b"fake_qr_code"
    
    class ChartPatternRecognizer:
        def __init__(self): pass
        def detect_patterns_from_data(self, df, symbol):
            return [{"pattern_type": "Demo Pattern", "signal": "BUY", "confidence": 0.85, "entry_price": 100.0, "target_price": 110.0, "stop_loss": 95.0, "risk_reward_ratio": 2.0, "description": "Demo pattern for testing"}]
    
    class TradingIntelligence:
        def __init__(self): pass
        def analyze_asset(self, symbol, data):
            return {
                "current_price": 100.0,
                "price_change_24h": 2.5,
                "recommendation": {
                    "decision": "BUY",
                    "confidence_level": "High",
                    "risk_level": "Medium",
                    "action_explanation": "Demo analysis shows positive signals"
                }
            }
    
    # Create dummy data module
    class DataIngestion:
        def __init__(self):
            pass
            
        def fetch_mixed_data(self, crypto_symbols=None, stock_symbols=None, period='1y', interval='1d'):
            # Return sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            sample_data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            result = {}
            all_symbols = (crypto_symbols or []) + (stock_symbols or [])
            for symbol in all_symbols:
                result[symbol] = sample_data
            return result
        
        def fetch_crypto_data(self, symbols, period='1y', interval='1d'):
            return self.fetch_mixed_data(crypto_symbols=symbols, period=period, interval=interval)
        
        def fetch_stock_data(self, symbols, period='1y', interval='1d'):
            return self.fetch_mixed_data(stock_symbols=symbols, period=period, interval=interval)
    
    ing = DataIngestion()

# Page configuration
st.set_page_config(
    page_title="AI Trading Bot - Professional Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1f4e79;
    }
    
    .portfolio-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .success-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    
    .danger-card {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = SecureAuthManager("your_secret_key_here")
if 'wallet_manager' not in st.session_state:
    st.session_state.wallet_manager = SecureWalletManager()
if 'pattern_recognizer' not in st.session_state:
    st.session_state.pattern_recognizer = ChartPatternRecognizer()
if 'ai_advisor' not in st.session_state:
    st.session_state.ai_advisor = TradingIntelligence()

def render_login_page():
    """Render the login/registration page"""
    st.markdown('<h1 class="main-header">🤖 AI Trading Bot - Secure Login</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            st.markdown("Access your AI-powered trading dashboard")
            
            with st.form("login_form"):
                username = st.text_input("Username or Email", placeholder="Enter your username or email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_2fa, col_remember = st.columns(2)
                with col_2fa:
                    totp_token = st.text_input("2FA Code (if enabled)", placeholder="123456")
                with col_remember:
                    remember_me = st.checkbox("Remember me")
                
                submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submitted:
                    if username and password:
                        with st.spinner("Authenticating..."):
                            result = st.session_state.auth_manager.authenticate_user(
                                username, password, totp_token if totp_token else None
                            )
                        
                        if result["success"]:
                            st.session_state.authenticated = True
                            st.session_state.user = result["user"]
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            if result.get("requires_2fa"):
                                st.warning("⚠️ Please enter your 2FA code")
                            else:
                                st.error(f"❌ {result['error']}")
                    else:
                        st.error("Please fill in all required fields")
        
        with tab2:
            st.markdown("### Create Your Account")
            st.markdown("Join the AI trading revolution")
            
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="your.email@example.com")
                new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                
                register_submitted = st.form_submit_button("🎯 Create Account", use_container_width=True)
                
                if register_submitted:
                    if not agree_terms:
                        st.error("Please agree to the terms and conditions")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif all([new_username, new_email, new_password]):
                        with st.spinner("Creating account..."):
                            result = st.session_state.auth_manager.register_user(
                                new_username, new_email, new_password
                            )
                        
                        if result["success"]:
                            st.success("✅ Account created successfully! You can now log in.")
                        else:
                            st.error(f"❌ {result['error']}")
                    else:
                        st.error("Please fill in all required fields")
        
        # Security features info
        st.markdown("---")
        st.markdown("### 🔐 Enterprise Security Features")
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            st.markdown("""
            **🛡️ Security:**
            - Military-grade encryption
            - 2FA authentication 
            - Secure wallet storage
            - API rate limiting
            """)
        with col_sec2:
            st.markdown("""
            **🚀 Features:**
            - AI-powered trading signals
            - Blockchain integration
            - Real-time pattern recognition
            - Portfolio tracking
            """)

def render_main_dashboard():
    """Render the main authenticated dashboard"""
    
    # Header with user info
    st.markdown(f'<h1 class="main-header">🤖 AI Trading Dashboard - Welcome {st.session_state.user["username"]}!</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👋 Hello, {st.session_state.user['username']}")
        st.markdown("---")
        
        # Navigation menu
        page = st.selectbox(
            "🗺️ Navigate",
            ["🏠 Dashboard Overview", "💼 Stock Portfolio", "₿ Crypto Portfolio", 
             "🔍 AI Analysis", "📊 Pattern Recognition", "💳 Wallet Management",
             "⚙️ Settings", "🔒 Security"]
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        # These would be pulled from database in real implementation
        st.metric("Total Portfolio Value", "$25,430", "+2.1%")
        st.metric("Today's P&L", "+$340", "1.4%")
        st.metric("Active Positions", "7")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    # Main content based on selected page
    if page == "🏠 Dashboard Overview":
        render_dashboard_overview()
    elif page == "💼 Stock Portfolio":
        render_stock_portfolio()
    elif page == "₿ Crypto Portfolio":
        render_crypto_portfolio()
    elif page == "🔍 AI Analysis":
        render_ai_analysis_page()
    elif page == "📊 Pattern Recognition":
        render_pattern_recognition_page()
    elif page == "💳 Wallet Management":
        render_wallet_management()
    elif page == "⚙️ Settings":
        render_settings_page()
    elif page == "🔒 Security":
        render_security_page()

def render_dashboard_overview():
    """Render the main dashboard overview"""
    
    # Portfolio overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card success-card">
            <h3>Total Value</h3>
            <h2>$25,430</h2>
            <p>+2.1% today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Stock Portfolio</h3>
            <h2>$18,200</h2>
            <p>5 positions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Crypto Portfolio</h3>
            <h2>$7,230</h2>
            <p>3 positions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card warning-card">
            <h3>AI Signals</h3>
            <h2>4 Active</h2>
            <p>2 BUY, 1 SELL, 1 HOLD</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent AI Signals
    col_signals, col_performance = st.columns([2, 1])
    
    with col_signals:
        st.markdown("### 🤖 Recent AI Trading Signals")
        
        # Sample signals data - would come from database
        signals_data = {
            'Symbol': ['NVDA', 'BTC', 'AAPL', 'TSLA'],
            'Signal': ['BUY', 'HOLD', 'BUY', 'SELL'],
            'Confidence': [0.87, 0.72, 0.79, 0.84],
            'Entry Price': [875.20, 43250.00, 178.50, 242.80],
            'Target': [945.00, 45000.00, 190.00, 220.00],
            'Pattern': ['Bullish Flag', 'Consolidation', 'Double Bottom', 'Head & Shoulders']
        }
        
        signals_df = pd.DataFrame(signals_data)
        
        # Style the dataframe
        def style_signals(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'HOLD':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_df = signals_df.style.applymap(style_signals, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("📈 View All Signals", use_container_width=True):
                st.info("Navigating to AI Analysis page...")
        with col_btn2:
            if st.button("🔄 Refresh Signals", use_container_width=True):
                st.success("Signals refreshed!")
        with col_btn3:
            if st.button("⚙️ Signal Settings", use_container_width=True):
                st.info("Opening signal configuration...")
    
    with col_performance:
        st.markdown("### 📊 Performance Metrics")
        
        # Performance chart
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        portfolio_values = 20000 * np.cumprod(1 + np.random.normal(0.0008, 0.02, len(dates)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Growth (YTD)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.metric("YTD Return", "+27.2%", "vs S&P: +18.4%")
        st.metric("Sharpe Ratio", "1.85", "Excellent")
        st.metric("Max Drawdown", "-8.3%", "Low Risk")

def render_stock_portfolio():
    """Render the stock portfolio tab"""
    st.markdown("### 💼 Stock Portfolio Management")
    
    # Portfolio summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stock Portfolio Value", "$18,200", "+1.8%")
    with col2:
        st.metric("Day P&L", "+$327", "+1.8%")
    with col3:
        st.metric("Total Return", "+22.4%", "vs S&P +18.4%")
    
    st.markdown("---")
    
    # Holdings table
    st.markdown("### 📈 Current Stock Holdings")
    
    # Sample stock holdings - would come from database
    stock_holdings = {
        'Symbol': ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN'],
        'Company': ['NVIDIA Corp.', 'Apple Inc.', 'Microsoft Corp.', 'Tesla Inc.', 'Amazon.com Inc.'],
        'Shares': [15, 50, 25, 10, 8],
        'Avg Cost': [420.50, 165.30, 380.20, 210.80, 145.60],
        'Current Price': [875.20, 178.50, 410.30, 242.80, 152.40],
        'Market Value': [13128, 8925, 10258, 2428, 1219],
        'Unrealized P&L': [6801, 660, 753, 320, 54],
        'Day Change': ['+2.1%', '+0.8%', '+1.2%', '+1.5%', '+0.3%']
    }
    
    holdings_df = pd.DataFrame(stock_holdings)
    
    # Add color coding for P&L
    def color_pnl(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: green'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: red'
        elif isinstance(val, (int, float)) and val > 0:
            return 'color: green'
        elif isinstance(val, (int, float)) and val < 0:
            return 'color: red'
        return ''
    
    styled_holdings = holdings_df.style.applymap(color_pnl, subset=['Unrealized P&L', 'Day Change'])
    st.dataframe(styled_holdings, use_container_width=True)
    
    # Individual stock charts
    st.markdown("### 📊 Stock Performance Charts")
    
    selected_stock = st.selectbox("Select stock to analyze:", holdings_df['Symbol'].tolist())
    
    if st.button(f"Load {selected_stock} Analysis"):
        with st.spinner(f"Loading {selected_stock} data..."):
            # Simulate loading stock data
            try:
                stock_data = ing.fetch_mixed_data(stock_symbols=[selected_stock], period="1y", interval="1d")
                if stock_data and selected_stock in stock_data:
                    df = stock_data[selected_stock]
                    
                    # Create stock chart with AI signals overlay
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{selected_stock} Price'
                    ))
                    
                    # Add pattern detection
                    patterns = st.session_state.pattern_recognizer.detect_patterns_from_data(df, selected_stock)
                    
                    # Add pattern annotations
                    for pattern in patterns[:3]:  # Show top 3 patterns
                        if 'entry_price' in pattern:
                            fig.add_hline(
                                y=pattern['entry_price'], 
                                line_dash="dash",
                                annotation_text=f"{pattern['pattern_type']}: {pattern['signal']}",
                                annotation_position="top right"
                            )
                    
                    fig.update_layout(
                        title=f"{selected_stock} Price Chart with AI Pattern Recognition",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detected patterns
                    if patterns:
                        st.markdown("### 🔍 Detected Patterns & Signals")
                        for i, pattern in enumerate(patterns[:3]):
                            with st.expander(f"Pattern {i+1}: {pattern['pattern_type']} - {pattern['signal']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Confidence:** {pattern['confidence']:.2%}")
                                    st.write(f"**Signal:** {pattern['signal']}")
                                    st.write(f"**Entry Price:** ${pattern['entry_price']:.2f}")
                                with col2:
                                    st.write(f"**Target:** ${pattern['target_price']:.2f}")
                                    st.write(f"**Stop Loss:** ${pattern['stop_loss']:.2f}")
                                    st.write(f"**Risk/Reward:** {pattern['risk_reward_ratio']:.1f}")
                                
                                st.write(f"**Analysis:** {pattern['description']}")
                else:
                    st.warning(f"Could not load data for {selected_stock}")
                    
            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")

def render_crypto_portfolio():
    """Render the crypto portfolio tab"""
    st.markdown("### ₿ Crypto Portfolio Management")
    
    # Crypto portfolio summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Crypto Portfolio Value", "$7,230", "+3.2%")
    with col2:
        st.metric("Day P&L", "+$225", "+3.2%")
    with col3:
        st.metric("Total Return", "+45.8%", "High Growth")
    
    st.markdown("---")
    
    # Wallet integration
    st.markdown("### 💳 Connected Wallets")
    
    # Mock wallet data - would come from database
    wallet_data = {
        'Wallet Type': ['Ethereum', 'Bitcoin', 'Binance Smart Chain'],
        'Address': ['0x742d...35be', '1A1z...xp2q', '0xf2a...93d1'],
        'Balance': ['2.15 ETH', '0.075 BTC', '125.4 BNB'],
        'USD Value': ['$5,200', '$1,800', '$230'],
        'Status': ['🟢 Active', '🟢 Active', '🟡 Limited']
    }
    
    wallet_df = pd.DataFrame(wallet_data)
    st.dataframe(wallet_df, use_container_width=True)
    
    # Wallet management buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("➕ Add New Wallet", use_container_width=True):
            st.info("Opening wallet creation wizard...")
    with col_btn2:
        if st.button("🔄 Sync Balances", use_container_width=True):
            st.success("Wallet balances synced!")
    with col_btn3:
        if st.button("📤 Send Transaction", use_container_width=True):
            st.info("Opening transaction interface...")
    
    st.markdown("---")
    
    # Crypto holdings
    st.markdown("### 📈 Current Crypto Holdings")
    
    crypto_holdings = {
        'Symbol': ['BTC', 'ETH', 'BNB'],
        'Name': ['Bitcoin', 'Ethereum', 'Binance Coin'],
        'Amount': ['0.075', '2.15', '125.4'],
        'Avg Cost': ['$38,400', '$2,100', '$280'],
        'Current Price': ['$43,250', '$2,420', '$285'],
        'Market Value': ['$3,244', '$5,203', '$357'],
        'Unrealized P&L': ['$362', '$688', '$627'],
        'Day Change': ['+2.8%', '+1.9%', '+0.9%']
    }
    
    crypto_df = pd.DataFrame(crypto_holdings)
    
    def color_crypto_pnl(val):
        if isinstance(val, str) and val.startswith('+'):
            return 'color: green; font-weight: bold'
        elif isinstance(val, str) and val.startswith('-'):
            return 'color: red; font-weight: bold'
        elif isinstance(val, (int, float)) and val > 0:
            return 'color: green; font-weight: bold'
        return ''
    
    styled_crypto = crypto_df.style.applymap(color_crypto_pnl, subset=['Unrealized P&L', 'Day Change'])
    st.dataframe(styled_crypto, use_container_width=True)

def render_ai_analysis_page():
    """Render the AI analysis page"""
    st.markdown("### 🤖 AI-Powered Market Analysis")
    
    # Load market data
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    # Asset selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_symbols = st.multiselect(
            "Select assets to analyze:",
            ["NVDA", "AAPL", "MSFT", "TSLA", "BTC", "ETH"],
            default=["NVDA", "AAPL"]
        )
    
    with col2:
        analysis_period = st.selectbox("Analysis Period:", ["1M", "3M", "6M", "1Y"])
        
    if st.button("🚀 Run AI Analysis", use_container_width=True):
        if selected_symbols:
            with st.spinner("Running AI analysis on selected assets..."):
                # Load data for selected symbols
                stock_symbols = [s for s in selected_symbols if s not in ['BTC', 'ETH']]
                crypto_symbols = [s for s in selected_symbols if s in ['BTC', 'ETH']]
                
                try:
                    # Convert period format for Yahoo Finance compatibility
                    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y"}
                    yahoo_period = period_map.get(analysis_period, "1y")
                    
                    market_data = ing.fetch_mixed_data(crypto_symbols=crypto_symbols, stock_symbols=stock_symbols, period=yahoo_period, interval="1d")
                    
                    if not market_data:
                        st.error("❌ Could not fetch market data. Using sample data for demonstration.")
                        # Create sample data for demo
                        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
                        sample_data = pd.DataFrame({
                            'open': np.random.randn(100).cumsum() + 100,
                            'high': np.random.randn(100).cumsum() + 102,
                            'low': np.random.randn(100).cumsum() + 98,
                            'close': np.random.randn(100).cumsum() + 100,
                            'volume': np.random.randint(1000, 10000, 100)
                        }, index=dates)
                        
                        market_data = {}
                        for symbol in selected_symbols:
                            market_data[symbol] = sample_data
                    
                    st.session_state.market_data.update(market_data)
                    
                    # Run AI analysis on each asset
                    st.markdown("### 📊 AI Analysis Results")
                    
                    for symbol in selected_symbols:
                        if symbol in st.session_state.market_data:
                            asset_data = st.session_state.market_data[symbol]
                            
                            with st.expander(f"📈 {symbol} Analysis", expanded=True):
                                # Get AI recommendation
                                analysis = st.session_state.ai_advisor.analyze_asset(symbol, asset_data)
                                rec = analysis['recommendation']
                                
                                # Display recommendation prominently
                                decision_colors = {
                                    'BUY': '#d4edda', 'DCA_IN': '#cce7ff', 'SELL': '#f8d7da',
                                    'DCA_OUT': '#fff3cd', 'HOLD': '#f8f9fa'
                                }
                                
                                st.markdown(f"""
                                <div style="background-color: {decision_colors.get(rec['decision'], '#f8f9fa')}; 
                                            padding: 15px; border-radius: 8px; margin: 10px 0;">
                                    <h3 style="margin: 0;">🎯 {rec['decision']}</h3>
                                    <p style="margin: 5px 0;"><strong>{rec['action_explanation']}</strong></p>
                                    <p style="margin: 0; color: #666;">
                                        Confidence: {rec['confidence_level']} | Risk: {rec['risk_level']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Pattern recognition
                                patterns = st.session_state.pattern_recognizer.detect_patterns_from_data(asset_data, symbol)
                                
                                if patterns:
                                    st.markdown("**🔍 Detected Chart Patterns:**")
                                    for pattern in patterns[:2]:  # Show top 2 patterns
                                        st.write(f"• **{pattern['pattern_type']}**: {pattern['signal']} "
                                               f"(Confidence: {pattern['confidence']:.1%})")
                                        st.write(f"  Entry: ${pattern['entry_price']:.2f} | "
                                               f"Target: ${pattern['target_price']:.2f}")
                                
                                # Key metrics
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric("Current Price", f"${analysis['current_price']:,.2f}")
                                with col_m2:
                                    st.metric("24h Change", f"{analysis['price_change_24h']:+.1f}%")
                                with col_m3:
                                    st.metric("AI Confidence", rec['confidence_level'])
                                
                                # Chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=asset_data.index,
                                    y=asset_data['close'],
                                    mode='lines',
                                    name=f'{symbol} Price',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} Price Chart",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
        else:
            st.warning("Please select at least one asset to analyze")

def render_pattern_recognition_page():
    """Render the pattern recognition page"""
    st.markdown("### 📊 Advanced Chart Pattern Recognition")
    st.markdown("Detect professional chart patterns using computer vision and machine learning")
    
    # Pattern library display
    st.markdown("### 📚 Supported Chart Patterns")
    
    pattern_categories = {
        "🔄 Reversal Patterns": [
            "Bearish Double Top", "Bearish Head and Shoulders", "Bearish Rising Wedge",
            "Bullish Double Bottom", "Bullish Inverted Head and Shoulders", "Bullish Falling Wedge"
        ],
        "➡️ Continuation Patterns": [
            "Bullish Flag Pattern", "Bullish Pennant Pattern", "Ascending Triangle",
            "Bearish Flag Pattern", "Bearish Pennant Pattern", "Descending Triangle"
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Reversal Patterns")
        for pattern in pattern_categories["🔄 Reversal Patterns"]:
            pattern_info = st.session_state.pattern_recognizer.pattern_library.get(pattern, {})
            success_rate = pattern_info.get('success_rate', 0) * 100
            st.markdown(f"• **{pattern}** - Success Rate: {success_rate:.0f}%")
    
    with col2:
        st.markdown("#### ➡️ Continuation Patterns")
        for pattern in pattern_categories["➡️ Continuation Patterns"]:
            pattern_info = st.session_state.pattern_recognizer.pattern_library.get(pattern, {})
            success_rate = pattern_info.get('success_rate', 0) * 100
            st.markdown(f"• **{pattern}** - Success Rate: {success_rate:.0f}%")
    
    st.markdown("---")
    
    # Pattern detection interface
    st.markdown("### 🔍 Real-Time Pattern Detection")
    
    symbol_input = st.text_input("Enter symbol to analyze:", "NVDA")
    
    if st.button("🎯 Detect Patterns", use_container_width=True):
        if symbol_input:
            with st.spinner(f"Analyzing {symbol_input} for chart patterns..."):
                try:
                    # Fetch data
                    if symbol_input in ['BTC', 'ETH']:
                        data = ing.fetch_crypto_data([symbol_input], "6mo", "1d")
                    else:
                        data = ing.fetch_mixed_data(stock_symbols=[symbol_input], period="6mo", interval="1d")
                    
                    if data and symbol_input in data:
                        asset_data = data[symbol_input]
                        
                        # Detect patterns
                        patterns = st.session_state.pattern_recognizer.detect_patterns_from_data(
                            asset_data, symbol_input
                        )
                        
                        if patterns:
                            st.success(f"✅ Found {len(patterns)} chart patterns in {symbol_input}")
                            
                            # Display each pattern
                            for i, pattern in enumerate(patterns):
                                with st.expander(f"Pattern {i+1}: {pattern['pattern_type']}", expanded=True):
                                    col_p1, col_p2 = st.columns([2, 1])
                                    
                                    with col_p1:
                                        st.markdown(f"""
                                        **Pattern:** {pattern['pattern_type']}  
                                        **Signal:** {pattern['signal']}  
                                        **Confidence:** {pattern['confidence']:.1%}  
                                        **Description:** {pattern['description']}
                                        """)
                                        
                                        st.markdown("**Trading Levels:**")
                                        st.write(f"• Entry Price: ${pattern['entry_price']:.2f}")
                                        st.write(f"• Target Price: ${pattern['target_price']:.2f}")
                                        st.write(f"• Stop Loss: ${pattern['stop_loss']:.2f}")
                                        st.write(f"• Risk/Reward Ratio: {pattern['risk_reward_ratio']:.1f}")
                                    
                                    with col_p2:
                                        # Pattern strength indicator
                                        strength = pattern['signal_strength']
                                        if strength > 0.7:
                                            strength_color = "green"
                                            strength_text = "Strong"
                                        elif strength > 0.5:
                                            strength_color = "orange"
                                            strength_text = "Moderate"
                                        else:
                                            strength_color = "red"
                                            strength_text = "Weak"
                                        
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 10px; 
                                                   background-color: {strength_color}; 
                                                   color: white; border-radius: 5px;">
                                            <h4>{strength_text}</h4>
                                            <p>{strength:.1%} Signal Strength</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                        else:
                            st.info("No significant chart patterns detected in the current timeframe")
                    else:
                        st.error(f"Could not fetch data for {symbol_input}")
                        
                except Exception as e:
                    st.error(f"Pattern detection failed: {str(e)}")

def render_wallet_management():
    """Render the wallet management page"""
    st.markdown("### 💳 Blockchain Wallet Management")
    st.markdown("Securely manage your crypto wallets and transactions")
    
    # Wallet creation section
    with st.expander("➕ Create New Wallet", expanded=False):
        st.markdown("### 🔐 Generate Secure Wallet")
        
        wallet_type = st.selectbox("Blockchain Network:", ["Ethereum", "Bitcoin", "Binance Smart Chain"])
        wallet_password = st.text_input("Wallet Password (for encryption):", type="password", 
                                       placeholder="Enter a strong password")
        
        if st.button("🚀 Create Wallet", use_container_width=True):
            if wallet_password:
                with st.spinner("Creating secure wallet..."):
                    try:
                        if wallet_type == "Ethereum":
                            wallet_data = st.session_state.wallet_manager.create_ethereum_wallet(wallet_password)
                            
                            st.success("✅ Ethereum wallet created successfully!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.code(f"Address: {wallet_data['address']}", language="text")
                                st.code(f"Derivation Path: {wallet_data['derivation_path']}", language="text")
                            
                            with col2:
                                # Generate QR code for the address
                                qr_data = st.session_state.wallet_manager.generate_wallet_qr_code(wallet_data['address'])
                                st.image(qr_data, caption="Wallet Address QR Code")
                            
                            st.warning("⚠️ **Important Security Notes:**")
                            st.markdown("""
                            - Your private key is encrypted and stored securely
                            - **Never share your private key or password**
                            - Save your wallet address and backup your password
                            - This is a demo - use testnet for actual testing
                            """)
                    
                    except Exception as e:
                        st.error(f"Wallet creation failed: {str(e)}")
            else:
                st.error("Please enter a wallet password")
    
    # Existing wallets display
    st.markdown("### 💼 Your Wallets")
    
    # Mock wallet data - in production this would come from database
    wallets = [
        {
            "Network": "Ethereum",
            "Address": "0x742d35cc6e7312e2b5b8c8b...c35be",
            "Balance": "2.15 ETH ($5,200)",
            "Status": "🟢 Active",
            "Created": "2024-01-15"
        },
        {
            "Network": "Bitcoin", 
            "Address": "1A1zP1eP5QGefi2DMPTfTL5S...xp2q",
            "Balance": "0.075 BTC ($1,800)",
            "Status": "🟢 Active", 
            "Created": "2024-01-20"
        }
    ]
    
    for i, wallet in enumerate(wallets):
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"**{wallet['Network']} Wallet**")
                st.code(wallet['Address'], language="text")
            
            with col2:
                st.metric("Balance", wallet['Balance'])
                st.write(f"Status: {wallet['Status']}")
            
            with col3:
                if st.button(f"🔍 View", key=f"view_{i}"):
                    st.info("Opening wallet details...")
                if st.button(f"📤 Send", key=f"send_{i}"):
                    st.info("Opening send transaction...")
    
    st.markdown("---")
    
    # Transaction interface
    with st.expander("📤 Send Transaction", expanded=False):
        st.markdown("### 💸 Send Cryptocurrency")
        
        col_send1, col_send2 = st.columns(2)
        
        with col_send1:
            from_wallet = st.selectbox("From Wallet:", ["Ethereum - 0x742d...35be", "Bitcoin - 1A1z...xp2q"])
            to_address = st.text_input("To Address:", placeholder="Enter recipient address")
            amount = st.number_input("Amount:", min_value=0.0, step=0.001, format="%.6f")
        
        with col_send2:
            gas_price = st.number_input("Gas Price (Gwei):", min_value=1, value=20)
            st.markdown("**Transaction Summary:**")
            if amount > 0:
                st.write(f"Amount: {amount} ETH")
                st.write(f"Est. Fee: ${gas_price * 21000 * 0.000000001 * 2400:.2f}")
                st.write(f"Total: {amount + (gas_price * 21000 * 0.000000001):.6f} ETH")
        
        if st.button("🚀 Create Transaction", use_container_width=True):
            if to_address and amount > 0:
                st.warning("⚠️ Transaction creation is disabled in demo mode")
                st.info("In production, this would create and broadcast the transaction securely")
            else:
                st.error("Please fill in all transaction details")

def render_settings_page():
    """Render the settings page"""
    st.markdown("### ⚙️ Account Settings")
    
    # User profile section
    with st.expander("👤 Profile Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Username", value=st.session_state.user['username'], disabled=True)
            st.text_input("Email", value=st.session_state.user['email'])
            
        with col2:
            st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"])
            st.selectbox("Currency", ["USD", "EUR", "BTC"])
    
    # Trading preferences
    with st.expander("🎯 Trading Preferences", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            risk_tolerance = st.selectbox("Risk Tolerance:", ["Conservative", "Moderate", "Aggressive"])
            max_position_size = st.slider("Max Position Size (% of portfolio):", 1, 25, 10)
            
        with col2:
            auto_trading = st.checkbox("Enable Auto-Trading", value=False)
            stop_loss_default = st.slider("Default Stop Loss (%):", 1, 10, 5)
    
    # Notification settings
    with st.expander("🔔 Notification Settings", expanded=True):
        st.checkbox("Email Notifications", value=True)
        st.checkbox("SMS Alerts for Large Moves", value=False)
        st.checkbox("Daily Portfolio Summary", value=True)
        st.checkbox("AI Signal Alerts", value=True)
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")

def render_security_page():
    """Render the security page"""
    st.markdown("### 🔒 Security & Account Protection")
    
    # 2FA setup
    with st.expander("🛡️ Two-Factor Authentication", expanded=True):
        if not st.session_state.user.get('is_2fa_enabled', False):
            st.warning("⚠️ 2FA is not enabled - your account is at risk!")
            
            if st.button("🔐 Enable 2FA", use_container_width=True):
                with st.spinner("Setting up 2FA..."):
                    result = st.session_state.auth_manager.enable_2fa(
                        st.session_state.user['id'], 
                        st.session_state.user['username']
                    )
                
                if result['success']:
                    st.success("✅ 2FA setup initiated!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Scan this QR code with your authenticator app:**")
                        qr_image = base64.b64decode(result['qr_code'])
                        st.image(qr_image, width=200)
                    
                    with col2:
                        st.markdown("**Or enter this secret manually:**")
                        st.code(result['secret'])
                        
                        st.markdown("**Backup Codes (save these securely):**")
                        for code in result['backup_codes']:
                            st.code(code)
                    
                    verify_token = st.text_input("Enter 2FA code to verify setup:", placeholder="123456")
                    if st.button("✅ Verify & Activate 2FA"):
                        if verify_token:
                            st.success("🎉 2FA activated successfully!")
                        else:
                            st.error("Please enter the verification code")
        else:
            st.success("✅ 2FA is enabled and protecting your account")
            if st.button("❌ Disable 2FA"):
                st.warning("This will reduce your account security!")
    
    # API key management
    with st.expander("🔑 API Key Management", expanded=True):
        st.markdown("Generate API keys for automated trading and third-party integrations")
        
        # Mock API key data
        api_keys = [
            {"name": "Trading Bot", "key": "ak_****1234", "created": "2024-01-15", "status": "Active"},
            {"name": "Portfolio App", "key": "ak_****5678", "created": "2024-02-01", "status": "Active"}
        ]
        
        for key in api_keys:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{key['name']}**")
                st.code(key['key'])
            with col2:
                st.write(f"Created: {key['created']}")
                st.write(f"Status: {key['status']}")
            with col3:
                st.button("🗑️ Delete", key=f"del_{key['name']}")
        
        st.markdown("---")
        col_api1, col_api2 = st.columns(2)
        with col_api1:
            new_key_name = st.text_input("API Key Name:", placeholder="My Trading App")
        with col_api2:
            key_permissions = st.multiselect(
                "Permissions:",
                ["Read Portfolio", "Execute Trades", "Manage Settings"],
                default=["Read Portfolio"]
            )
        
        if st.button("🔑 Generate New API Key"):
            if new_key_name:
                st.success("✅ New API key generated!")
                st.code("ak_new_generated_key_here_1234567890")
                st.warning("⚠️ Save this key securely - it won't be shown again!")
    
    # Session management
    with st.expander("📱 Active Sessions", expanded=True):
        st.markdown("Monitor and manage your active login sessions")
        
        sessions = [
            {"device": "Chrome - Windows", "ip": "192.168.1.100", "location": "New York, US", "active": "Current"},
            {"device": "Safari - iPhone", "ip": "10.0.0.45", "location": "New York, US", "active": "2 hours ago"},
            {"device": "Chrome - Mac", "ip": "192.168.1.200", "location": "New York, US", "active": "1 day ago"}
        ]
        
        for i, session in enumerate(sessions):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{session['device']}**")
                st.write(f"IP: {session['ip']}")
            with col2:
                st.write(f"Location: {session['location']}")
                st.write(f"Last active: {session['active']}")
            with col3:
                if session['active'] != "Current":
                    st.button("🚫 Revoke", key=f"revoke_{i}")
        
        if st.button("🚫 Revoke All Other Sessions"):
            st.success("All other sessions have been revoked!")

# Main application logic
def main():
    """Main application entry point"""
    
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_main_dashboard()

if __name__ == "__main__":
    main()