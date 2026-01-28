"""
Live Trade Feed - Real-time trade notifications with visual cards
Shows each trade as it happens with explanation, P&L, and stop loss details
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

try:
    from trading.auto_trader import get_auto_trader, reset_auto_trader, AutoTrader
    AUTO_TRADER_AVAILABLE = True
except ImportError:
    AUTO_TRADER_AVAILABLE = False

def inject_trade_feed_styles():
    """Inject CSS for trade notification cards"""
    st.markdown("""
    <style>
    .trade-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .trade-card.buy {
        border-left-color: #00d4aa;
        background: linear-gradient(135deg, #0a3d2e 0%, #1a1a2e 100%);
    }
    
    .trade-card.sell {
        border-left-color: #ff6b6b;
        background: linear-gradient(135deg, #3d1a1a 0%, #1a1a2e 100%);
    }
    
    .trade-card.closed {
        border-left-color: #a855f7;
        background: linear-gradient(135deg, #2d1a3d 0%, #1a1a2e 100%);
    }
    
    .trade-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .trade-symbol {
        font-size: 1.8em;
        font-weight: bold;
        color: #fff;
    }
    
    .trade-action {
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .trade-action.buy {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #000;
    }
    
    .trade-action.sell {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: #fff;
    }
    
    .trade-reason {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    .trade-reason-title {
        font-weight: bold;
        color: #667eea;
        margin-bottom: 8px;
        font-size: 0.9em;
        text-transform: uppercase;
    }
    
    .trade-metrics {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 15px 0;
    }
    
    .trade-metric {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.8em;
        color: #888;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #fff;
    }
    
    .metric-value.positive {
        color: #00d4aa;
    }
    
    .metric-value.negative {
        color: #ff6b6b;
    }
    
    .stop-loss-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px 15px;
        margin-top: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .stop-loss-label {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .take-profit-label {
        color: #00d4aa;
        font-weight: bold;
    }
    
    .trade-timestamp {
        font-size: 0.85em;
        color: #666;
        margin-top: 10px;
        text-align: right;
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,107,107,0.2);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        color: #ff6b6b;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #ff6b6b;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .no-trades-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        border: 2px dashed #333;
    }
    
    .no-trades-icon {
        font-size: 3em;
        margin-bottom: 15px;
    }
    
    .pnl-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
    }
    
    .pnl-item {
        text-align: center;
    }
    
    .pnl-label {
        font-size: 0.85em;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    
    .pnl-value {
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def generate_trade_explanation(trade: Dict) -> str:
    """Generate a beginner-friendly explanation of why a trade was made"""
    signal = trade.get('signal', 'BUY')
    symbol = trade.get('symbol', 'Unknown')
    confidence = trade.get('confidence', 75)
    
    reasons = trade.get('reasons', [])
    
    if not reasons:
        if signal == 'BUY':
            reasons = [
                f"The AI detected a strong upward momentum in {symbol}",
                f"Technical indicators suggest the price is likely to rise",
                f"Market conditions are favorable for buying"
            ]
        else:
            reasons = [
                f"The AI detected downward pressure on {symbol}",
                f"Technical indicators suggest selling is optimal",
                f"Taking profits to protect your gains"
            ]
    
    explanation = f"<strong>Why this trade?</strong><br>"
    explanation += f"The AI is {confidence}% confident in this decision because:<br>"
    explanation += "<ul style='margin: 10px 0; padding-left: 20px;'>"
    for reason in reasons[:3]:
        explanation += f"<li>{reason}</li>"
    explanation += "</ul>"
    
    return explanation


def render_trade_card(trade: Dict, show_closed: bool = False):
    """Render a single trade notification card using Streamlit components"""
    symbol = trade.get('symbol', 'Unknown')
    action = trade.get('action', trade.get('signal', 'BUY')).upper()
    entry_price = trade.get('entry_price', trade.get('price', 0))
    current_price = trade.get('current_price', entry_price)
    quantity = trade.get('quantity', trade.get('contracts', 1))
    stop_loss = trade.get('stop_loss', entry_price * 0.97)
    take_profit = trade.get('take_profit', entry_price * 1.06)
    confidence = trade.get('confidence', 75)
    timestamp = trade.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    status = trade.get('status', 'open')
    asset_type = trade.get('asset_type', 'stock').upper()
    
    if action in ['BUY', 'LONG']:
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:
        unrealized_pnl = (entry_price - current_price) * quantity
        unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
    
    potential_loss = abs(entry_price - stop_loss) * quantity
    potential_profit = abs(take_profit - entry_price) * quantity
    
    action_color = "#00d4aa" if action in ['BUY', 'LONG'] else "#ff6b6b"
    border_color = action_color
    pnl_color = "#00d4aa" if unrealized_pnl >= 0 else "#ff6b6b"
    
    reasons = trade.get('reasons', [])
    if not reasons:
        if action in ['BUY', 'LONG']:
            reasons = ["Strong momentum detected", "Technical indicators bullish", "Volume confirms trend"]
        else:
            reasons = ["Overbought conditions", "Taking profits", "Momentum weakening"]
    
    reasons_html = "".join([f"<li style='margin: 5px 0;'>{r}</li>" for r in reasons[:3]])
    
    with st.container():
        st.markdown(f"""<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; padding: 20px; margin: 15px 0; border-left: 5px solid {border_color}; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<div><span style="font-size: 1.8em; font-weight: bold; color: #fff;">{symbol}</span><span style="color: #888; font-size: 0.8em; margin-left: 10px;">{asset_type}</span></div>
<span style="padding: 8px 20px; border-radius: 20px; font-weight: bold; font-size: 1.1em; background: {action_color}; color: {'#000' if action in ['BUY', 'LONG'] else '#fff'};">{action}</span>
</div>
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin: 15px 0;">
<div style="font-weight: bold; color: #667eea; margin-bottom: 8px; font-size: 0.9em;">WHY THIS TRADE?</div>
<div style="color: #fff;">The AI is {confidence}% confident because:</div>
<ul style="margin: 10px 0; padding-left: 20px; color: rgba(255,255,255,0.9);">{reasons_html}</ul>
</div>
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
<div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; text-align: center;"><div style="font-size: 0.85em; color: #888; margin-bottom: 5px;">Entry Price</div><div style="font-size: 1.2em; font-weight: bold; color: #fff;">${entry_price:,.2f}</div></div>
<div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; text-align: center;"><div style="font-size: 0.85em; color: #888; margin-bottom: 5px;">Current Price</div><div style="font-size: 1.2em; font-weight: bold; color: #fff;">${current_price:,.2f}</div></div>
<div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; text-align: center;"><div style="font-size: 0.85em; color: #888; margin-bottom: 5px;">Quantity</div><div style="font-size: 1.2em; font-weight: bold; color: #fff;">{quantity:,.2f}</div></div>
<div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; text-align: center;"><div style="font-size: 0.85em; color: #888; margin-bottom: 5px;">Unrealized P&L</div><div style="font-size: 1.2em; font-weight: bold; color: {pnl_color};">${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.1f}%)</div></div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; margin: 15px 0; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
<div><span style="color: #ff6b6b; font-weight: bold;">🛑 Stop Loss:</span><span style="color: #fff; margin-left: 8px;">${stop_loss:,.2f}</span><span style="color: #888; font-size: 0.9em;"> (Risk: ${potential_loss:,.2f})</span></div>
<div><span style="color: #00d4aa; font-weight: bold;">🎯 Take Profit:</span><span style="color: #fff; margin-left: 8px;">${take_profit:,.2f}</span><span style="color: #888; font-size: 0.9em;"> (Potential: ${potential_profit:,.2f})</span></div>
</div>
<div style="color: #888; font-size: 0.9em; text-align: right;">⏰ {timestamp} | Confidence: {confidence}%</div>
</div>""", unsafe_allow_html=True)


def render_pnl_summary(trades: List[Dict], current_balance: float, starting_balance: float):
    """Render overall P&L summary"""
    open_trades = [t for t in trades if t.get('status') == 'open']
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    total_unrealized = sum(t.get('unrealized_pnl', 0) for t in open_trades)
    total_realized = sum(t.get('realized_pnl', 0) for t in closed_trades)
    total_pnl = current_balance - starting_balance
    win_rate = len([t for t in closed_trades if t.get('realized_pnl', 0) > 0]) / len(closed_trades) * 100 if closed_trades else 0
    
    html = f"""
    <div class="pnl-summary">
        <div class="pnl-item">
            <div class="pnl-label">💰 Total P&L</div>
            <div class="pnl-value" style="color: {'#00d4aa' if total_pnl >= 0 else '#ff6b6b'}">
                ${total_pnl:+,.2f}
            </div>
        </div>
        <div class="pnl-item">
            <div class="pnl-label">📈 Unrealized</div>
            <div class="pnl-value" style="color: {'#00d4aa' if total_unrealized >= 0 else '#ff6b6b'}">
                ${total_unrealized:+,.2f}
            </div>
        </div>
        <div class="pnl-item">
            <div class="pnl-label">✅ Realized</div>
            <div class="pnl-value" style="color: {'#00d4aa' if total_realized >= 0 else '#ff6b6b'}">
                ${total_realized:+,.2f}
            </div>
        </div>
        <div class="pnl-item">
            <div class="pnl-label">🎯 Win Rate</div>
            <div class="pnl-value">{win_rate:.1f}%</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_no_trades_message():
    """Show message when no trades have been executed"""
    auto_trading_active = st.session_state.get('auto_trading_active', False)
    
    if auto_trading_active:
        st.markdown("""
        <div class="no-trades-card">
            <div class="no-trades-icon">🤖</div>
            <h3 style="color: #fff; margin-bottom: 10px;">Auto-Trading is Active!</h3>
            <p style="color: #888; font-size: 1.1em;">
                Click <strong>"Scan for Trades"</strong> above to search for trading opportunities.
                The AI will analyze the market and execute trades when it finds good setups.
            </p>
            <p style="color: #00d4aa; margin-top: 15px;">
                💡 Tip: Each scan checks multiple assets and executes trades automatically when conditions are right.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-trades-card">
            <div class="no-trades-icon">⏳</div>
            <h3 style="color: #fff; margin-bottom: 10px;">Waiting for Trade Signals...</h3>
            <p style="color: #888; font-size: 1.1em;">
                The AI is analyzing the market. Trades will appear here as soon as 
                the system identifies profitable opportunities.
            </p>
            <p style="color: #667eea; margin-top: 15px;">
                💡 Tip: The AI looks for high-confidence setups that match your risk profile.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_auto_trading_stats(stats: Dict[str, Any]):
    """Render auto-trading statistics panel"""
    pnl_color = "#00d4aa" if stats['total_pnl'] >= 0 else "#ff6b6b"
    pnl_sign = "+" if stats['total_pnl'] >= 0 else ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 15px; margin-bottom: 20px; border: 1px solid #667eea33;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Balance</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #fff;">${stats['current_balance']:,.2f}</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Total P&L</div>
                <div style="font-size: 1.2em; font-weight: bold; color: {pnl_color};">{pnl_sign}${stats['total_pnl']:,.2f}</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Return</div>
                <div style="font-size: 1.2em; font-weight: bold; color: {pnl_color};">{pnl_sign}{stats['pnl_percent']:.1f}%</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Win Rate</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #fff;">{stats['win_rate']:.0f}%</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Trades</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #fff;">{stats['total_trades']}</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 100px;">
                <div style="font-size: 0.8em; color: #888;">Open</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #667eea;">{stats['open_positions']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_session_trades() -> List[Dict]:
    """Get trades from current trading session"""
    if 'trading_session' not in st.session_state:
        return []
    
    session = st.session_state.get('trading_session', {})
    trades = session.get('trades', [])
    
    if 'paper_trading_engine' in st.session_state:
        engine = st.session_state['paper_trading_engine']
        if hasattr(engine, 'session') and engine.session:
            positions = engine.session.positions
            for pos in positions:
                trade = {
                    'symbol': pos.symbol,
                    'action': pos.option_type.upper() if hasattr(pos, 'option_type') else 'BUY',
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'quantity': pos.contracts if hasattr(pos, 'contracts') else pos.quantity,
                    'stop_loss': pos.stop_loss if hasattr(pos, 'stop_loss') else pos.entry_price * 0.97,
                    'take_profit': pos.take_profit if hasattr(pos, 'take_profit') else pos.entry_price * 1.06,
                    'status': pos.status,
                    'unrealized_pnl': pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0,
                    'realized_pnl': pos.realized_pnl if hasattr(pos, 'realized_pnl') else 0,
                    'timestamp': pos.entry_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(pos, 'entry_time') else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'confidence': pos.confidence if hasattr(pos, 'confidence') else 75,
                    'asset_type': 'option' if hasattr(pos, 'option_type') else 'crypto'
                }
                trades.append(trade)
    
    return trades


def generate_demo_trade() -> Dict:
    """Generate a realistic demo trade for simulation"""
    import random
    
    symbols_data = {
        'AAPL': {'price': 185, 'name': 'Apple', 'type': 'stock'},
        'NVDA': {'price': 135, 'name': 'NVIDIA', 'type': 'stock'},
        'TSLA': {'price': 250, 'name': 'Tesla', 'type': 'stock'},
        'MSFT': {'price': 420, 'name': 'Microsoft', 'type': 'stock'},
        'GOOGL': {'price': 175, 'name': 'Google', 'type': 'stock'},
        'BTC-USD': {'price': 95000, 'name': 'Bitcoin', 'type': 'crypto'},
        'ETH-USD': {'price': 3200, 'name': 'Ethereum', 'type': 'crypto'},
        'SOL-USD': {'price': 180, 'name': 'Solana', 'type': 'crypto'},
    }
    
    reasons_buy = [
        "RSI dropped below 30 indicating oversold conditions",
        "Price bounced off 50-day moving average support",
        "Strong bullish divergence detected on MACD",
        "Volume spike with price breaking resistance level",
        "Golden cross pattern formed (50 MA crossing above 200 MA)",
        "Positive earnings surprise driving momentum"
    ]
    
    reasons_sell = [
        "RSI exceeded 70 indicating overbought conditions",
        "Price rejected at key resistance level",
        "Bearish divergence detected on momentum indicators",
        "Stop loss triggered to protect gains",
        "Take profit target reached",
        "Negative market sentiment shift detected"
    ]
    
    symbol = random.choice(list(symbols_data.keys()))
    symbol_info = symbols_data[symbol]
    action = random.choice(['BUY', 'SELL'])
    
    base_price = symbol_info['price']
    entry_price = base_price * random.uniform(0.98, 1.02)
    current_price = entry_price * random.uniform(0.97, 1.05)
    
    if symbol_info['type'] == 'crypto':
        quantity = round(random.uniform(0.01, 0.5), 4)
    else:
        quantity = random.randint(5, 50)
    
    confidence = random.randint(72, 94)
    
    return {
        'symbol': symbol,
        'action': action,
        'entry_price': round(entry_price, 2),
        'current_price': round(current_price, 2),
        'quantity': quantity,
        'stop_loss': round(entry_price * (0.97 if action == 'BUY' else 1.03), 2),
        'take_profit': round(entry_price * (1.06 if action == 'BUY' else 0.94), 2),
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'open',
        'asset_type': symbol_info['type'],
        'reasons': random.sample(reasons_buy if action == 'BUY' else reasons_sell, 3)
    }


def render_live_trade_feed():
    """Main function to render the live trade feed"""
    inject_trade_feed_styles()
    
    auto_trading_active = st.session_state.get('auto_trading_active', False)
    
    if auto_trading_active and not AUTO_TRADER_AVAILABLE:
        st.error("Auto-trading is not available. There may be a system configuration issue. Please try again later or contact support.")
        auto_trading_active = False
    
    if auto_trading_active and AUTO_TRADER_AVAILABLE:
        if 'auto_trader' not in st.session_state:
            auto_trader = get_auto_trader()
            wizard_config = st.session_state.get('trading_wizard_config', {})
            auto_trader.configure(
                risk_tolerance=wizard_config.get('risk_level', 'moderate'),
                asset_classes=wizard_config.get('asset_classes', ['stocks']),
                starting_capital=wizard_config.get('capital', 10000),
                timeframe=wizard_config.get('timeframe', 'day')
            )
            auto_trader.start()
            st.session_state['auto_trader'] = auto_trader
            st.session_state['last_scan_time'] = None
            st.session_state['last_scan_result'] = None
            st.session_state['initial_scan_done'] = False
        else:
            auto_trader = st.session_state['auto_trader']
        
        if not st.session_state.get('initial_scan_done', False) and auto_trader.is_active():
            with st.spinner("🔍 Running initial market scan..."):
                new_trades = auto_trader.run_scan()
                st.session_state['initial_scan_done'] = True
                st.session_state['last_scan_time'] = datetime.now()
                st.session_state['last_scan_result'] = len(new_trades)
                if new_trades:
                    for trade in new_trades:
                        st.toast(f"{'🟢' if trade['action'] == 'BUY' else '🔴'} New {trade['action']}: {trade['symbol']} @ ${trade['entry_price']:,.2f}", icon="📈")
    else:
        auto_trader = None
    
    is_running = auto_trader.is_active() if auto_trader else False
    status_indicator = "LIVE" if is_running else "PAUSED"
    status_color = "#00d4aa" if is_running else "#ffa500"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2 style="margin: 0;">📡 Live Trade Feed</h2>
        <div class="live-indicator" style="background: linear-gradient(135deg, {status_color}22 0%, {status_color}11 100%); border: 1px solid {status_color};">
            <div class="live-dot" style="background: {status_color};"></div>
            {status_indicator}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'demo_trades' not in st.session_state:
        st.session_state['demo_trades'] = []
    
    if auto_trader:
        trades = auto_trader.get_all_trades()
        stats = auto_trader.get_stats()
        starting_balance = stats['starting_balance']
        current_balance = stats['current_balance']
    else:
        trades = get_session_trades()
        if not trades:
            trades = st.session_state.get('demo_trades', [])
        starting_balance = st.session_state.get('starting_balance', 10000)
        current_balance = st.session_state.get('current_balance', starting_balance)
    
    if auto_trader and is_running:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("🔍 Scan for Trades", use_container_width=True, type="primary"):
                with st.spinner("Scanning markets..."):
                    new_trades = auto_trader.run_scan()
                    st.session_state['last_scan_time'] = datetime.now()
                    st.session_state['last_scan_result'] = len(new_trades)
                    if new_trades:
                        for trade in new_trades:
                            st.toast(f"{'🟢' if trade['action'] == 'BUY' else '🔴'} New {trade['action']}: {trade['symbol']} @ ${trade['entry_price']:,.2f}", icon="📈")
                    else:
                        st.toast("No opportunities found - market conditions not ideal. Try again in a few minutes!", icon="👀")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Prices", use_container_width=True):
                auto_trader.update_positions()
                st.rerun()
        
        with col3:
            if st.button("⏸️ Pause Trading", use_container_width=True):
                auto_trader.stop()
                st.toast("Auto-trading paused", icon="⏸️")
                st.rerun()
        
        with col4:
            if st.button("🛑 Stop & Exit", use_container_width=True):
                auto_trader.stop()
                st.session_state['auto_trading_active'] = False
                st.session_state.pop('auto_trader', None)
                reset_auto_trader()
                st.toast("Auto-trading stopped", icon="🛑")
                st.rerun()
    elif auto_trader and not is_running:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("▶️ Resume Trading", use_container_width=True, type="primary"):
                auto_trader.start()
                st.toast("Auto-trading resumed!", icon="▶️")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Prices", use_container_width=True):
                auto_trader.update_positions()
                st.rerun()
        
        with col3:
            if st.button("🛑 Stop & Exit", use_container_width=True):
                st.session_state['auto_trading_active'] = False
                st.session_state.pop('auto_trader', None)
                reset_auto_trader()
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("➕ Simulate New Trade", use_container_width=True, type="primary"):
                new_trade = generate_demo_trade()
                st.session_state['demo_trades'].insert(0, new_trade)
                st.toast(f"{'🟢' if new_trade['action'] == 'BUY' else '🔴'} New {new_trade['action']} trade: {new_trade['symbol']} @ ${new_trade['entry_price']:,.2f}", icon="📈")
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Prices", use_container_width=True):
                import random
                for trade in st.session_state.get('demo_trades', []):
                    if trade.get('status') == 'open':
                        change = random.uniform(-0.02, 0.03)
                        trade['current_price'] = round(trade['entry_price'] * (1 + change), 2)
                st.rerun()
        
        with col3:
            if st.session_state.get('demo_trades') and st.button("🗑️ Clear All", use_container_width=True):
                st.session_state['demo_trades'] = []
                st.rerun()
    
    st.markdown("---")
    
    if auto_trader:
        last_scan_time = st.session_state.get('last_scan_time')
        last_scan_result = st.session_state.get('last_scan_result', 0)
        
        if last_scan_time:
            time_ago = datetime.now() - last_scan_time
            if time_ago.seconds < 60:
                time_str = f"{time_ago.seconds} seconds ago"
            elif time_ago.seconds < 3600:
                time_str = f"{time_ago.seconds // 60} minutes ago"
            else:
                time_str = last_scan_time.strftime('%H:%M:%S')
            
            result_text = f"Found {last_scan_result} trading opportunities" if last_scan_result > 0 else "No opportunities found (market conditions not ideal)"
            result_color = "#00d4aa" if last_scan_result > 0 else "#888"
            
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); border-radius: 8px; padding: 10px 15px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #888;">📡 Last scan: <strong style="color: #fff;">{time_str}</strong></span>
                <span style="color: {result_color};">{result_text}</span>
            </div>
            """, unsafe_allow_html=True)
        
        stats = auto_trader.get_stats()
        render_auto_trading_stats(stats)
    
    all_trades = trades if trades else st.session_state.get('demo_trades', [])
    
    if all_trades:
        render_pnl_summary(all_trades, current_balance, starting_balance)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            show_open = st.checkbox("Show Open Trades", value=True)
        with col2:
            show_closed = st.checkbox("Show Closed Trades", value=True)
        
        open_trades = [t for t in all_trades if t.get('status') == 'open']
        closed_trades = [t for t in all_trades if t.get('status') == 'closed']
        
        if show_open and open_trades:
            st.markdown("### 🔵 Open Positions")
            for i, trade in enumerate(open_trades):
                render_trade_card(trade)
                col_close, col_space = st.columns([1, 3])
                with col_close:
                    if st.button(f"Close Position", key=f"close_{i}", use_container_width=True):
                        trade['status'] = 'closed'
                        if trade['action'] == 'BUY':
                            trade['realized_pnl'] = (trade['current_price'] - trade['entry_price']) * trade['quantity']
                        else:
                            trade['realized_pnl'] = (trade['entry_price'] - trade['current_price']) * trade['quantity']
                        st.rerun()
        
        if show_closed and closed_trades:
            st.markdown("### ✅ Closed Trades")
            for trade in closed_trades:
                render_trade_card(trade, show_closed=True)
        
        if not (show_open and open_trades) and not (show_closed and closed_trades):
            render_no_trades_message()
    else:
        render_no_trades_message()
        st.info("💡 Click 'Simulate New Trade' above to see how trades will appear when the AI executes them.")


def add_trade_notification(trade_data: Dict):
    """Add a new trade notification to the session"""
    if 'trade_notifications' not in st.session_state:
        st.session_state['trade_notifications'] = []
    
    notification = {
        **trade_data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'id': len(st.session_state['trade_notifications']) + 1
    }
    
    st.session_state['trade_notifications'].insert(0, notification)
    
    if len(st.session_state['trade_notifications']) > 50:
        st.session_state['trade_notifications'] = st.session_state['trade_notifications'][:50]
    
    return notification


def render_trade_notification_popup(trade: Dict):
    """Render a popup-style notification for a new trade"""
    symbol = trade.get('symbol', 'Unknown')
    action = trade.get('action', 'BUY')
    price = trade.get('entry_price', trade.get('price', 0))
    
    st.toast(f"{'🟢' if action == 'BUY' else '🔴'} New {action} trade: {symbol} @ ${price:,.2f}", icon="📈")
