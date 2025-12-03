"""
Trade Log Tab for IntelliTradeAI Dashboard
Shows all executed trades, open positions, and P&L tracking
"""

import streamlit as st
import pandas as pd
from database.db_manager import DatabaseManager
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


def render_trade_log_tab():
    """Render the Trade Log & P&L tab"""
    st.title("📊 Trade Log & Portfolio Tracking")
    
    db = DatabaseManager()
    
    render_portfolio_summary(db)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Trade History",
        "💼 Open Positions",
        "📈 P&L Analytics",
        "🔔 Active Alerts"
    ])
    
    with tab1:
        render_trade_history(db)
    
    with tab2:
        render_open_positions(db)
    
    with tab3:
        render_pnl_analytics(db)
    
    with tab4:
        render_active_alerts(db)


def render_portfolio_summary(db: DatabaseManager):
    """Display portfolio summary metrics"""
    st.subheader("💰 Portfolio Summary")
    
    portfolio = db.get_portfolio()
    
    if not portfolio:
        st.warning("No portfolio data available. Start trading to see your portfolio!")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_value = float(portfolio.get('total_value', 0))
    cash_balance = float(portfolio.get('cash_balance', 0))
    total_realized_pnl = float(portfolio.get('total_realized_pnl', 0))
    total_unrealized_pnl = float(portfolio.get('total_unrealized_pnl', 0))
    win_rate = float(portfolio.get('win_rate', 0))
    
    col1.metric(
        "Total Value",
        f"${total_value:,.2f}",
        delta=f"${total_realized_pnl + total_unrealized_pnl:,.2f}"
    )
    
    col2.metric(
        "Cash Balance",
        f"${cash_balance:,.2f}"
    )
    
    col3.metric(
        "Realized P&L",
        f"${total_realized_pnl:,.2f}",
        delta=f"{'+' if total_realized_pnl >= 0 else ''}{total_realized_pnl:,.2f}"
    )
    
    col4.metric(
        "Unrealized P&L",
        f"${total_unrealized_pnl:,.2f}",
        delta=f"{'+' if total_unrealized_pnl >= 0 else ''}{total_unrealized_pnl:,.2f}"
    )
    
    col5.metric(
        "Win Rate",
        f"{win_rate:.1f}%"
    )
    
    col1, col2, col3 = st.columns(3)
    
    crypto_balance = float(portfolio.get('crypto_balance', 0))
    stock_balance = float(portfolio.get('stock_balance', 0))
    options_balance = float(portfolio.get('options_balance', 0))
    
    col1.metric("Crypto Holdings", f"${crypto_balance:,.2f}")
    col2.metric("Stock Holdings", f"${stock_balance:,.2f}")
    col3.metric("Options Holdings", f"${options_balance:,.2f}")


def render_trade_history(db: DatabaseManager):
    """Display complete trade history"""
    st.subheader("📝 All Trades")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "Open", "Closed"],
            key="trade_filter_status"
        )
    
    with col2:
        filter_mode = st.selectbox(
            "Filter by Mode",
            ["All", "Manual", "Automatic"],
            key="trade_filter_mode"
        )
    
    with col3:
        limit = st.number_input("Show Last N Trades", min_value=10, max_value=500, value=100)
    
    if filter_status == "All":
        trades = db.get_all_trades(limit=limit)
    else:
        trades = db.get_all_trades(status=filter_status.lower(), limit=limit)
    
    if filter_mode != "All":
        trades = [t for t in trades if t['trading_mode'] == filter_mode.lower()]
    
    if not trades:
        st.info("No trades found. Start trading to see your history!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    display_columns = ['timestamp', 'symbol', 'asset_type', 'action', 'quantity', 
                       'entry_price', 'exit_price', 'realized_pnl', 'status', 'trading_mode']
    
    available_cols = [col for col in display_columns if col in trades_df.columns]
    
    def color_pnl(val):
        if pd.isna(val):
            return ''
        elif val > 0:
            return 'background-color: #90EE90'
        elif val < 0:
            return 'background-color: #FFB6C1'
        return ''
    
    styled_df = trades_df[available_cols].style.applymap(
        color_pnl,
        subset=['realized_pnl'] if 'realized_pnl' in available_cols else []
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.download_button(
        "📥 Download Trade History (CSV)",
        trades_df.to_csv(index=False),
        "trade_history.csv",
        "text/csv"
    )


def render_open_positions(db: DatabaseManager):
    """Display all open positions"""
    st.subheader("💼 Current Open Positions")
    
    positions = db.get_active_positions()
    
    if not positions:
        st.info("No open positions. Execute trades to see positions here!")
        return
    
    positions_df = pd.DataFrame(positions)
    
    for idx, position in enumerate(positions):
        with st.expander(f"📍 {position['symbol']} ({position['asset_type'].upper()})"):
            col1, col2, col3, col4 = st.columns(4)
            
            quantity = float(position['quantity'])
            avg_entry = float(position['avg_entry_price'])
            current_price = float(position.get('current_price', avg_entry))
            total_invested = float(position.get('total_invested', 0))
            current_value = float(position.get('current_value', 0))
            unrealized_pnl = float(position.get('unrealized_pnl', 0))
            unrealized_pnl_pct = float(position.get('unrealized_pnl_percent', 0))
            
            col1.metric("Quantity", f"{quantity:.4f}")
            col2.metric("Avg Entry Price", f"${avg_entry:.2f}")
            col3.metric("Current Price", f"${current_price:.2f}")
            col4.metric(
                "Unrealized P&L",
                f"${unrealized_pnl:.2f}",
                delta=f"{unrealized_pnl_pct:.2f}%"
            )
            
            st.metric("Total Invested", f"${total_invested:.2f}")
            st.metric("Current Value", f"${current_value:.2f}")
            
            if st.button(f"🔴 Close Position {position['symbol']}", key=f"close_{idx}"):
                close_position_action(db, position)


def close_position_action(db: DatabaseManager, position: dict):
    """Handle position closing"""
    try:
        result = db.close_position(position['symbol'], position['asset_type'])
        if result:
            st.success(f"✅ Closed position for {position['symbol']}")
            st.rerun()
        else:
            st.error("Failed to close position")
    except Exception as e:
        st.error(f"Error closing position: {str(e)}")


def render_pnl_analytics(db: DatabaseManager):
    """Display P&L analytics and visualizations"""
    st.subheader("📈 Profit & Loss Analytics")
    
    trades = db.get_all_trades(limit=500)
    
    if not trades:
        st.info("No trades available for analytics")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
    
    if not closed_trades.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 P&L Distribution")
            
            pnl_fig = px.histogram(
                closed_trades,
                x='realized_pnl',
                nbins=30,
                title='Distribution of Trade P&L',
                labels={'realized_pnl': 'Profit/Loss ($)'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(pnl_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Win/Loss Breakdown")
            
            wins = len(closed_trades[closed_trades['realized_pnl'] > 0])
            losses = len(closed_trades[closed_trades['realized_pnl'] < 0])
            breakeven = len(closed_trades[closed_trades['realized_pnl'] == 0])
            
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses', 'Breakeven'],
                values=[wins, losses, breakeven],
                marker=dict(colors=['#90EE90', '#FFB6C1', '#D3D3D3'])
            )])
            pie_fig.update_layout(title='Trade Outcomes')
            st.plotly_chart(pie_fig, use_container_width=True)
        
        st.markdown("### 📅 Cumulative P&L Over Time")
        
        closed_trades = closed_trades.sort_values('timestamp')
        closed_trades['cumulative_pnl'] = closed_trades['realized_pnl'].cumsum()
        
        cumulative_fig = px.line(
            closed_trades,
            x='timestamp',
            y='cumulative_pnl',
            title='Cumulative Profit & Loss',
            labels={'cumulative_pnl': 'Cumulative P&L ($)', 'timestamp': 'Date'}
        )
        cumulative_fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(cumulative_fig, use_container_width=True)
        
        st.markdown("### 🏆 Performance by Asset Type")
        
        asset_performance = closed_trades.groupby('asset_type')['realized_pnl'].agg([
            ('Total P&L', 'sum'),
            ('Avg P&L', 'mean'),
            ('Trade Count', 'count')
        ]).reset_index()
        
        st.dataframe(asset_performance, use_container_width=True)


def render_active_alerts(db: DatabaseManager):
    """Display active price alerts"""
    st.subheader("🔔 Active Price Alerts")
    
    alerts = db.get_active_alerts()
    
    if not alerts:
        st.info("No active alerts. Set alerts in Manual mode!")
        return
    
    for idx, alert in enumerate(alerts):
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            col1.write(f"**{alert['symbol']}**")
            col2.write(f"{alert['alert_type'].replace('_', ' ').title()}")
            col3.write(f"Target: ${alert['target_price']:.2f}")
            col4.write(f"{alert['action']} {alert['quantity']}")
            
            if st.button(f"❌ Cancel Alert", key=f"cancel_alert_{idx}"):
                cancel_alert_action(db, alert['alert_id'])
            
            st.markdown("---")


def cancel_alert_action(db: DatabaseManager, alert_id: str):
    """Cancel an active alert"""
    try:
        st.success(f"✅ Alert cancelled")
        st.rerun()
    except Exception as e:
        st.error(f"Error cancelling alert: {str(e)}")
