"""
Paper Trading Dashboard for IntelliTradeAI
Options-focused paper trading with real-time monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.paper_trading_engine import PaperTradingEngine, TARGET_SYMBOLS, StrategyConfig

st.set_page_config(page_title="Paper Trading - IntelliTradeAI", layout="wide")

if 'paper_engine' not in st.session_state:
    st.session_state.paper_engine = PaperTradingEngine(
        starting_balance=100000.0,
        target_balance=200000.0,
        max_drawdown=30.0
    )

if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

engine = st.session_state.paper_engine

st.title("Options Paper Trading")
st.markdown("Test AI predictions with simulated options trading on 12 target stocks")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Target Symbols", len(TARGET_SYMBOLS))
with col2:
    st.metric("Starting Balance", "$100,000")
with col3:
    st.metric("Target Balance", "$200,000")
with col4:
    st.metric("Max Drawdown", "30%")

st.subheader("Target Stocks")
st.write(" | ".join(TARGET_SYMBOLS))

st.markdown("---")

col_control, col_status = st.columns([1, 2])

with col_control:
    st.subheader("Session Control")
    
    if not engine.session or engine.session.status != 'active':
        if st.button("Start New Session", type="primary", use_container_width=True):
            engine.start_session()
            st.success("Paper trading session started!")
            st.rerun()
    else:
        st.success(f"Session Active: {engine.session.session_id[:8]}...")
        
        if st.button("Run Trading Cycle", type="primary", use_container_width=True):
            with st.spinner("Executing trading cycle..."):
                result = engine.run_trading_cycle()
                
                if result.get('status', {}).get('status') == 'target_reached':
                    st.balloons()
                    st.success("TARGET REACHED! You've doubled your money!")
                elif result.get('status', {}).get('status') == 'drawdown_exceeded':
                    st.error("Drawdown limit exceeded. Analyze and restart.")
                else:
                    st.info(f"Cycle complete: {result.get('trades_executed', 0)} trades executed")
            st.rerun()
        
        if st.button("End Session", type="secondary", use_container_width=True):
            engine._end_session('manual_stop')
            st.warning("Session ended manually.")
            st.rerun()
        
        if st.button("Restart with Improvements", use_container_width=True):
            engine.restart_with_improvements()
            st.success("Session restarted with improved strategy!")
            st.rerun()

with col_status:
    st.subheader("Session Status")
    
    if engine.session:
        summary = engine.get_session_summary()
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            portfolio_value = summary.get('portfolio_value', 100000)
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                delta=f"${portfolio_value - 100000:,.2f}"
            )
        
        with metrics_col2:
            return_pct = summary.get('return_pct', 0)
            st.metric(
                "Return",
                f"{return_pct:.2f}%",
                delta=f"{return_pct:.2f}%"
            )
        
        with metrics_col3:
            drawdown = summary.get('current_drawdown', 0)
            st.metric(
                "Current Drawdown",
                f"{drawdown:.2f}%",
                delta=f"-{drawdown:.2f}%" if drawdown > 0 else "0%",
                delta_color="inverse"
            )
        
        with metrics_col4:
            win_rate = summary.get('win_rate', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{summary.get('winning_trades', 0)}/{summary.get('total_trades', 0)} trades"
            )
        
        progress = ((portfolio_value - 100000) / 100000) * 100
        st.progress(min(100, max(0, progress / 100)), text=f"Progress to Target: {progress:.1f}%")
        
        drawdown_color = "green" if drawdown < 15 else ("orange" if drawdown < 25 else "red")
        st.progress(min(100, drawdown / 30), text=f"Drawdown: {drawdown:.1f}% / 30%")
    else:
        st.info("No active session. Start a new session to begin paper trading.")

st.markdown("---")

if engine.session:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Open Positions", "Trade History", "Performance", "AI Signals", "Strategy"
    ])
    
    with tab1:
        st.subheader("Open Positions")
        
        open_positions = [p for p in engine.session.positions if p.status == 'open']
        
        if open_positions:
            pos_data = []
            for p in open_positions:
                entry_value = p.entry_price * p.contracts * 100
                current_value = p.current_price * p.contracts * 100
                pnl_pct = ((current_value - entry_value) / entry_value * 100) if entry_value > 0 else 0
                
                pos_data.append({
                    'Symbol': p.symbol,
                    'Type': p.option_type.upper(),
                    'Strike': f"${p.strike_price:,.2f}",
                    'Expiry': p.expiration_date,
                    'Contracts': str(p.contracts),
                    'Entry': f"${p.entry_price:.2f}",
                    'Current': f"${p.current_price:.2f}",
                    'P&L': f"${p.unrealized_pnl:,.2f}",
                    'P&L %': f"{pnl_pct:.1f}%",
                    'Delta': f"{p.current_delta:.3f}",
                    'Theta': f"{p.current_theta:.3f}",
                    'AI Signal': f"{p.ai_signal} ({p.ai_confidence:.0f}%)"
                })
            
            df = pd.DataFrame(pos_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            total_delta = sum(p.current_delta * p.contracts * 100 for p in open_positions)
            total_theta = sum(p.current_theta * p.contracts * 100 for p in open_positions)
            total_unrealized = sum(p.unrealized_pnl for p in open_positions)
            
            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                st.metric("Portfolio Delta", f"{total_delta:.2f}")
            with gcol2:
                st.metric("Portfolio Theta", f"${total_theta:.2f}/day")
            with gcol3:
                st.metric("Total Unrealized P&L", f"${total_unrealized:,.2f}")
        else:
            st.info("No open positions. Run a trading cycle to generate trades.")
    
    with tab2:
        st.subheader("Trade History")
        
        if engine.session.trade_history:
            trade_data = []
            for t in reversed(engine.session.trade_history[-50:]):
                trade_data.append({
                    'Time': t.get('executed_at', '')[:19],
                    'Symbol': t.get('symbol', ''),
                    'Action': t.get('action', ''),
                    'Type': t.get('option_type', '').upper(),
                    'Strike': f"${t.get('strike', 0):,.2f}",
                    'Contracts': str(t.get('contracts', 0)),
                    'Price': f"${t.get('price', 0):.2f}",
                    'Total': f"${t.get('total_value', 0):,.2f}",
                    'P&L': f"${t.get('realized_pnl', 0):,.2f}" if 'realized_pnl' in t else '-',
                    'Reason': t.get('close_reason', t.get('ai_signal', '-'))
                })
            
            df = pd.DataFrame(trade_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet. Run trading cycles to generate trade history.")
    
    with tab3:
        st.subheader("Performance Analytics")
        
        summary = engine.get_session_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trading Statistics**")
            stats_data = {
                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 
                          'Total Realized P&L', 'Total Unrealized P&L', 'Strategy Version'],
                'Value': [
                    str(summary.get('total_trades', 0)),
                    str(summary.get('winning_trades', 0)),
                    str(summary.get('losing_trades', 0)),
                    f"{summary.get('win_rate', 0):.1f}%",
                    f"${summary.get('total_realized_pnl', 0):,.2f}",
                    f"${summary.get('total_unrealized_pnl', 0):,.2f}",
                    f"v{summary.get('strategy_version', 1)}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Portfolio Breakdown**")
            portfolio_value = summary.get('portfolio_value', 100000)
            cash = summary.get('current_balance', 100000)
            positions_value = portfolio_value - cash
            
            fig = go.Figure(data=[go.Pie(
                labels=['Cash', 'Options Positions'],
                values=[cash, positions_value],
                hole=0.4,
                marker_colors=['#2ecc71', '#3498db']
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if engine.session.snapshots:
            st.markdown("**Portfolio Value Over Time**")
            snapshot_data = pd.DataFrame(engine.session.snapshots)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(snapshot_data))),
                y=snapshot_data['portfolio_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#3498db', width=2)
            ))
            fig.add_hline(y=100000, line_dash="dash", line_color="gray", annotation_text="Start")
            fig.add_hline(y=200000, line_dash="dash", line_color="green", annotation_text="Target")
            fig.update_layout(
                height=300,
                xaxis_title="Trading Cycle",
                yaxis_title="Portfolio Value ($)",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Current AI Signals")
        
        if st.button("Refresh Signals", use_container_width=False):
            with st.spinner("Analyzing markets..."):
                signals = engine.get_ai_signals()
                st.session_state.current_signals = signals
        
        if 'current_signals' in st.session_state and st.session_state.current_signals:
            signals = st.session_state.current_signals
            
            for signal in signals:
                with st.expander(f"{signal['symbol']} - {signal['signal']} ({signal['confidence']:.0f}%)"):
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    with scol1:
                        st.metric("Signal", signal['signal'])
                    with scol2:
                        st.metric("Confidence", f"{signal['confidence']:.0f}%")
                    with scol3:
                        st.metric("Price", f"${signal['current_price']:.2f}")
                    with scol4:
                        st.metric("RSI", f"{signal['rsi']:.1f}")
                    
                    st.write(f"5-Day Change: {signal['pct_change_5d']:.2f}%")
                    st.write(f"SMA(20): ${signal['sma_20']:.2f}")
        else:
            st.info("Click 'Refresh Signals' to analyze current market conditions.")
    
    with tab5:
        st.subheader("Strategy Configuration")
        
        st.markdown(f"**Current Strategy Version: v{engine.strategy.version}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Position Management**")
            st.write(f"- Position Size: {engine.strategy.position_size_percent}% of portfolio")
            st.write(f"- Max Positions: {engine.strategy.max_positions}")
            st.write(f"- Min Confidence: {engine.strategy.min_confidence}%")
        
        with col2:
            st.markdown("**Risk Management**")
            st.write(f"- Stop Loss: {engine.strategy.stop_loss_percent}%")
            st.write(f"- Take Profit: {engine.strategy.take_profit_percent}%")
            st.write(f"- Delta Range: {engine.strategy.delta_range_min} - {engine.strategy.delta_range_max}")
        
        st.markdown("**Options Parameters**")
        st.write(f"- Days to Expiry: {engine.strategy.min_days_to_expiry} - {engine.strategy.max_days_to_expiry}")
        st.write(f"- Allowed Symbols: {', '.join(engine.strategy.allowed_symbols)}")
        
        if engine.session.improvements:
            st.markdown("---")
            st.subheader("Improvement History")
            
            for imp in engine.session.improvements:
                with st.expander(f"Improvement v{imp.get('new_strategy_version', '?')} - {imp.get('trigger_reason', '')}"):
                    changes = imp.get('improvements_made', {})
                    for change in changes.get('strategy_changes', []):
                        st.write(f"- {change}")

else:
    st.markdown("---")
    st.info("Start a paper trading session to see positions, trades, and performance analytics.")

st.markdown("---")
st.caption("Paper Trading Mode - No real money at risk. Results are simulated.")
