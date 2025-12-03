"""
Options Analysis Tab for IntelliTradeAI Dashboard
Provides options chain viewer, AI analysis, and manual/automatic trading
"""

import streamlit as st
import pandas as pd
from data.options_data_fetcher import OptionsDataFetcher
from trading.mode_manager import TradingModeManager, TradingMode
from trading.trade_executor import TradeExecutor
from database.db_manager import DatabaseManager
from datetime import datetime


def render_options_analysis_tab():
    """Render the Options Analysis tab"""
    st.title("📊 Options Analysis & Trading")
    
    db = DatabaseManager()
    mode_manager = st.session_state.get('mode_manager')
    if not mode_manager:
        mode_manager = TradingModeManager(TradingMode.MANUAL)
        st.session_state['mode_manager'] = mode_manager
    
    trade_executor = TradeExecutor(db, mode_manager)
    options_fetcher = OptionsDataFetcher()
    
    render_trading_mode_selector(mode_manager)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Options Chain Explorer")
        
        stock_symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="Enter a stock ticker to view its options chain"
        ).upper()
        
        if st.button("📡 Fetch Options Chain", type="primary"):
            with st.spinner(f"Fetching options data for {stock_symbol}..."):
                chain_data = options_fetcher.fetch_options_chain(stock_symbol)
                st.session_state['options_chain'] = chain_data
    
    with col2:
        st.subheader("⚙️ Analysis Settings")
        
        strategy = st.selectbox(
            "Strategy Type",
            ["conservative", "moderate", "aggressive"],
            index=1,
            help="Select risk tolerance level"
        )
        
        show_greeks = st.checkbox("Show Greeks", value=True)
        show_recommendations = st.checkbox("Show AI Recommendations", value=True)
    
    if 'options_chain' in st.session_state:
        chain_data = st.session_state['options_chain']
        
        if 'error' in chain_data:
            st.error(f"❌ {chain_data['error']}")
        else:
            render_options_chain_display(chain_data, show_greeks)
            
            if show_recommendations:
                st.markdown("---")
                render_ai_recommendations(options_fetcher, stock_symbol, strategy, trade_executor)


def render_trading_mode_selector(mode_manager: TradingModeManager):
    """Render trading mode selector"""
    st.subheader("🎛️ Trading Mode")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        current_mode = mode_manager.get_current_mode()
        mode_display = "🚗 MANUAL" if current_mode == TradingMode.MANUAL else "🤖 AUTOMATIC"
        st.metric("Current Mode", mode_display)
    
    with col2:
        if st.button("🚗 Switch to Manual", use_container_width=True):
            result = mode_manager.switch_mode(TradingMode.MANUAL)
            st.success(result['message'])
            st.rerun()
    
    with col3:
        if st.button("🤖 Switch to Automatic", use_container_width=True):
            result = mode_manager.switch_mode(TradingMode.AUTOMATIC)
            st.success(result['message'])
            st.rerun()
    
    with st.expander("ℹ️ Mode Information"):
        config = mode_manager.get_active_config()
        st.write(f"**{config['mode'].upper()} Mode**: {config['description']}")
        
        if config['mode'] == 'automatic':
            st.write("**Configuration:**")
            st.json(config['config'])
        else:
            st.write("**Configuration:**")
            st.json(config['config'])


def render_options_chain_display(chain_data: dict, show_greeks: bool):
    """Display options chain data"""
    st.subheader(f"📈 {chain_data['symbol']} Options Chain")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${chain_data['current_price']:.2f}")
    col2.metric("Expiration", chain_data['expiration_date'])
    col3.metric("Available Expirations", len(chain_data['available_expirations']))
    
    tab1, tab2 = st.tabs(["📞 CALLS", "📉 PUTS"])
    
    with tab1:
        calls_df = chain_data['calls']
        if not calls_df.empty:
            display_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                               'openInterest', 'impliedVolatility', 'inTheMoney']
            
            if show_greeks and 'delta' in calls_df.columns:
                display_columns.extend(['delta', 'gamma', 'theta', 'vega'])
            
            available_cols = [col for col in display_columns if col in calls_df.columns]
            
            st.dataframe(
                calls_df[available_cols].style.background_gradient(
                    subset=['volume'] if 'volume' in available_cols else None,
                    cmap='Greens'
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No call options available")
    
    with tab2:
        puts_df = chain_data['puts']
        if not puts_df.empty:
            display_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                               'openInterest', 'impliedVolatility', 'inTheMoney']
            
            if show_greeks and 'delta' in puts_df.columns:
                display_columns.extend(['delta', 'gamma', 'theta', 'vega'])
            
            available_cols = [col for col in display_columns if col in puts_df.columns]
            
            st.dataframe(
                puts_df[available_cols].style.background_gradient(
                    subset=['volume'] if 'volume' in available_cols else None,
                    cmap='Reds'
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No put options available")


def render_ai_recommendations(options_fetcher: OptionsDataFetcher, symbol: str, 
                                strategy: str, trade_executor: TradeExecutor):
    """Render AI-powered options recommendations"""
    st.subheader("🤖 AI Options Recommendations")
    
    with st.spinner("Analyzing options strategies..."):
        recommendations = options_fetcher.find_optimal_options(symbol, strategy)
    
    if 'error' in recommendations:
        st.error(recommendations['error'])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📞 CALL Recommendations")
        call_recs = recommendations.get('call_recommendations', [])
        
        if call_recs:
            for i, rec in enumerate(call_recs[:3]):
                with st.container():
                    st.markdown(f"**Option #{i+1}**")
                    st.write(f"**Strike:** ${rec['strike']:.2f}")
                    st.write(f"**Premium:** ${rec['premium']:.2f}")
                    st.write(f"**Delta:** {rec.get('delta', 0):.4f}")
                    st.write(f"**Breakeven:** ${rec.get('breakeven', 0):.2f}")
                    st.info(f"💡 {rec['reason']}")
                    
                    if st.button(f"Trade this Call #{i+1}", key=f"call_{i}"):
                        execute_option_trade(trade_executor, symbol, 'CALL', rec, recommendations['current_price'])
                    
                    st.markdown("---")
        else:
            st.info("No call recommendations available")
    
    with col2:
        st.markdown("### 📉 PUT Recommendations")
        put_recs = recommendations.get('put_recommendations', [])
        
        if put_recs:
            for i, rec in enumerate(put_recs[:3]):
                with st.container():
                    st.markdown(f"**Option #{i+1}**")
                    st.write(f"**Strike:** ${rec['strike']:.2f}")
                    st.write(f"**Premium:** ${rec['premium']:.2f}")
                    st.write(f"**Delta:** {rec.get('delta', 0):.4f}")
                    st.write(f"**Breakeven:** ${rec.get('breakeven', 0):.2f}")
                    st.info(f"💡 {rec['reason']}")
                    
                    if st.button(f"Trade this Put #{i+1}", key=f"put_{i}"):
                        execute_option_trade(trade_executor, symbol, 'PUT', rec, recommendations['current_price'])
                    
                    st.markdown("---")
        else:
            st.info("No put recommendations available")


def execute_option_trade(trade_executor: TradeExecutor, symbol: str, option_type: str, 
                          recommendation: dict, current_price: float):
    """Execute an options trade"""
    st.session_state['pending_option_trade'] = {
        'symbol': symbol,
        'option_type': option_type,
        'strike': recommendation['strike'],
        'premium': recommendation['premium'],
        'current_price': current_price,
        'recommendation': recommendation
    }
    
    st.info(f"Trade details loaded. Configure quantity and confirm below.")


if 'pending_option_trade' in st.session_state:
    st.markdown("---")
    st.subheader("✅ Confirm Options Trade")
    
    trade_details = st.session_state['pending_option_trade']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Symbol:** {trade_details['symbol']}")
        st.write(f"**Type:** {trade_details['option_type']}")
        st.write(f"**Strike:** ${trade_details['strike']:.2f}")
        st.write(f"**Premium:** ${trade_details['premium']:.2f}")
    
    with col2:
        quantity = st.number_input("Number of Contracts", min_value=1, max_value=100, value=1)
        total_cost = trade_details['premium'] * quantity * 100
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Confirm Trade", type="primary", use_container_width=True):
            db = DatabaseManager()
            mode_manager = st.session_state.get('mode_manager', TradingModeManager())
            trade_executor = TradeExecutor(db, mode_manager)
            
            trade_params = {
                'symbol': trade_details['symbol'],
                'action': 'BUY',
                'quantity': quantity,
                'asset_type': 'option',
                'current_price': trade_details['premium'],
                'signal': {
                    'action': 'BUY',
                    'confidence': 75,
                    'option_type': trade_details['option_type'],
                    'strike': trade_details['strike']
                }
            }
            
            result = trade_executor.execute_trade(trade_params)
            
            if result.get('success'):
                st.success(f"✅ {result.get('message', 'Trade executed successfully!')}")
                del st.session_state['pending_option_trade']
                st.rerun()
            else:
                st.error(f"❌ {result.get('message', 'Trade failed')}")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            del st.session_state['pending_option_trade']
            st.rerun()
