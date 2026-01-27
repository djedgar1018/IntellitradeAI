"""
Beginner Learning Module for IntelliTradeAI
Educational content to help new investors understand trading concepts
"""

import streamlit as st


def render_learning_hub():
    """Render the beginner learning hub page"""
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>🎓 Trading Learning Hub</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">
            Master trading concepts at your own pace
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Trading Basics", 
        "📊 Understanding Signals", 
        "⚠️ Risk Management",
        "🤖 How the AI Works"
    ])
    
    with tab1:
        render_trading_basics()
    
    with tab2:
        render_signals_explained()
    
    with tab3:
        render_risk_management()
    
    with tab4:
        render_ai_explainer()


def render_trading_basics():
    """Explain basic trading concepts"""
    
    st.markdown("## 📚 Trading Basics for Beginners")
    
    with st.expander("💡 What is Trading?", expanded=True):
        st.markdown("""
        **Trading** is buying and selling financial assets (like stocks, crypto, or currencies) 
        to make a profit from price changes.
        
        **Example:** You buy 10 shares of Apple at $150 each ($1,500 total). 
        Later, the price rises to $165. You sell and make $150 profit!
        
        **Key Terms:**
        - **Buy (Long):** Purchasing an asset hoping it will go UP
        - **Sell (Short):** Selling an asset hoping it will go DOWN
        - **Position:** An active trade you currently hold
        - **Profit/Loss (P&L):** How much you've made or lost
        """)
    
    with st.expander("📊 Types of Assets"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Stocks**
            - Ownership shares in companies
            - Trade during market hours (9:30am-4pm ET)
            - Examples: Apple, Tesla, Microsoft
            
            **₿ Cryptocurrency**
            - Digital currencies on blockchain
            - Trades 24/7, every day
            - Examples: Bitcoin, Ethereum
            """)
        
        with col2:
            st.markdown("""
            **💱 Forex (Currencies)**
            - Trading currency pairs
            - Massive market, very liquid
            - Examples: EUR/USD, GBP/USD
            
            **📑 Options**
            - Contracts giving right to buy/sell
            - Higher risk, higher reward
            - For more advanced traders
            """)
    
    with st.expander("⏰ Trading Timeframes"):
        st.markdown("""
        **How long you hold a trade matters:**
        
        | Style | Hold Time | Trades/Day | Best For |
        |-------|-----------|------------|----------|
        | Scalping | Minutes | 10-50 | Full-time traders |
        | Day Trading | Hours | 3-10 | Active traders |
        | Swing Trading | Days-Weeks | 1-5/week | Part-time traders |
        | Position Trading | Weeks-Months | 1-4/month | Long-term investors |
        
        **Our Recommendation:** Swing Trading is great for beginners - enough action to learn, 
        but not overwhelming!
        """)


def render_signals_explained():
    """Explain AI signals in simple terms"""
    
    st.markdown("## 📊 Understanding AI Trading Signals")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    ">
        <h3 style="margin: 0 0 12px 0;">What are Signals?</h3>
        <p style="margin: 0; opacity: 0.9;">
            Signals are the AI's recommendations based on analyzing thousands of data points. 
            Think of them as a smart friend who's really good at spotting patterns in the market.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            height: 200px;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">📈</div>
            <h3>BUY Signal</h3>
            <p style="font-size: 0.9em; opacity: 0.9;">
                "This looks like a good time to purchase this asset. 
                The AI sees potential for price to go UP."
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            height: 200px;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">📉</div>
            <h3>SELL Signal</h3>
            <p style="font-size: 0.9em; opacity: 0.9;">
                "Consider selling or avoiding this asset. 
                The AI sees potential for price to go DOWN."
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            height: 200px;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">⏸️</div>
            <h3>HOLD Signal</h3>
            <p style="font-size: 0.9em; opacity: 0.9;">
                "Wait for a clearer opportunity. 
                The market is uncertain right now."
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Understanding Confidence Levels")
    
    st.markdown("""
    The AI gives each signal a **confidence score** (0-100%):
    
    | Confidence | What it Means | Action |
    |------------|---------------|--------|
    | 80-100% | Very confident | Strong signal, consider acting |
    | 60-79% | Moderately confident | Good signal, proceed with caution |
    | 40-59% | Uncertain | Weak signal, maybe wait |
    | 0-39% | Low confidence | AI is unsure, don't rely on this |
    
    **Pro Tip:** Focus on signals with 70%+ confidence when starting out!
    """)


def render_risk_management():
    """Explain risk management concepts"""
    
    st.markdown("## ⚠️ Risk Management - Protecting Your Money")
    
    st.warning("""
    **The #1 Rule of Trading:** Never risk more than you can afford to lose!
    
    Even the best traders have losing trades. What separates winners from losers 
    is how they manage risk.
    """)
    
    with st.expander("🛑 Stop Loss - Your Safety Net", expanded=True):
        st.markdown("""
        A **Stop Loss** automatically sells your position if the price drops too much.
        
        **Example:**
        - You buy stock at $100
        - You set a stop loss at $95 (5% below)
        - If price drops to $95, it automatically sells
        - Maximum loss: $5 per share (5%)
        
        **Why Use Stop Loss:**
        - Limits your maximum loss per trade
        - Removes emotion from decisions
        - Protects you from big drops while you're away
        
        **Our AI automatically sets stop losses for all trades!**
        """)
    
    with st.expander("🎯 Take Profit - Locking in Gains"):
        st.markdown("""
        A **Take Profit** automatically sells when you've made enough profit.
        
        **Example:**
        - You buy stock at $100
        - You set take profit at $110 (10% above)
        - If price rises to $110, it automatically sells
        - You lock in $10 profit per share!
        
        **Why Use Take Profit:**
        - Ensures you capture gains before reversals
        - "Profit is profit" - don't get greedy!
        """)
    
    with st.expander("📊 Position Sizing - How Much to Invest"):
        st.markdown("""
        **Never put all your eggs in one basket!**
        
        | Risk Level | Max Per Trade | Why |
        |------------|---------------|-----|
        | Conservative | 1-2% of portfolio | Very safe, slow growth |
        | Moderate | 3-5% of portfolio | Balanced approach |
        | Aggressive | 5-10% of portfolio | Faster growth, more risk |
        
        **Example with $10,000 portfolio:**
        - Conservative: Max $100-200 per trade
        - Moderate: Max $300-500 per trade
        - Aggressive: Max $500-1000 per trade
        
        **This way, even 5 losing trades won't wipe you out!**
        """)


def render_ai_explainer():
    """Explain how the AI trading system works"""
    
    st.markdown("## 🤖 How the AI Trading Assistant Works")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    ">
        <h3 style="margin: 0 0 16px 0;">🧠 Think of the AI as Your Expert Advisor</h3>
        <p style="color: rgba(255,255,255,0.8);">
            It analyzes millions of data points in seconds - something no human could do. 
            It looks at price patterns, news, social sentiment, and technical indicators 
            to find the best trading opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 What the AI Analyzes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technical Analysis:**
        - 📈 Price patterns (trends, support, resistance)
        - 📊 Volume (buying/selling pressure)
        - 🔄 Moving averages (trend direction)
        - 📉 RSI, MACD (overbought/oversold)
        
        **Market Sentiment:**
        - 😊 Fear & Greed Index
        - 📰 News headlines and tone
        - 🐦 Social media buzz
        """)
    
    with col2:
        st.markdown("""
        **Pattern Recognition:**
        - 🔷 Chart patterns (triangles, flags)
        - 📊 Historical similarities
        - 🔮 Machine learning predictions
        
        **Risk Assessment:**
        - ⚠️ Market volatility levels
        - 📉 Potential downside
        - 🎯 Risk/reward ratios
        """)
    
    st.markdown("### 🔄 The AI Decision Process")
    
    st.markdown("""
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;">
            <div style="
                background: #667eea;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 8px auto;
                font-size: 24px;
            ">1</div>
            <strong>Collect Data</strong>
            <p style="font-size: 0.8em; color: rgba(255,255,255,0.6);">Price, volume, news</p>
        </div>
        <div style="font-size: 24px; margin-top: 15px;">→</div>
        <div style="text-align: center;">
            <div style="
                background: #667eea;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 8px auto;
                font-size: 24px;
            ">2</div>
            <strong>Analyze</strong>
            <p style="font-size: 0.8em; color: rgba(255,255,255,0.6);">Run ML models</p>
        </div>
        <div style="font-size: 24px; margin-top: 15px;">→</div>
        <div style="text-align: center;">
            <div style="
                background: #667eea;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 8px auto;
                font-size: 24px;
            ">3</div>
            <strong>Generate Signal</strong>
            <p style="font-size: 0.8em; color: rgba(255,255,255,0.6);">BUY/SELL/HOLD</p>
        </div>
        <div style="font-size: 24px; margin-top: 15px;">→</div>
        <div style="text-align: center;">
            <div style="
                background: #11998e;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 8px auto;
                font-size: 24px;
            ">4</div>
            <strong>Execute</strong>
            <p style="font-size: 0.8em; color: rgba(255,255,255,0.6);">Place trade</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Remember:** The AI is a tool to help you, not a magic money printer. 
    Always understand why it's making recommendations and never invest more than you can afford to lose!
    """)
