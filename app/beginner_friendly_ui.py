"""
Beginner-Friendly UI Components for IntelliTradeAI
Makes trading accessible to everyone with clear visuals and simple explanations
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go


def inject_beginner_styles():
    """Inject modern, beginner-friendly CSS styles"""
    st.markdown("""
    <style>
    /* Modern Card Styles */
    .beginner-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .action-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 8px;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        text-align: center;
        color: white;
    }
    
    .action-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Signal Cards */
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
    }
    
    /* Icon Styles */
    .big-icon {
        font-size: 48px;
        margin-bottom: 12px;
    }
    
    /* Step Indicators */
    .step-indicator {
        display: flex;
        align-items: center;
        margin: 20px 0;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
    }
    
    .step-circle.completed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .step-circle.active {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.6); }
        50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.9); }
    }
    
    .step-line {
        flex: 1;
        height: 3px;
        background: rgba(255,255,255,0.2);
        margin: 0 10px;
    }
    
    .step-line.completed {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Info Boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    .warning-box {
        background: rgba(245, 166, 35, 0.1);
        border-left: 4px solid #f5a623;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    .success-box {
        background: rgba(17, 153, 142, 0.1);
        border-left: 4px solid #11998e;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    /* Tooltip Styles */
    .tooltip-trigger {
        color: #667eea;
        cursor: help;
        border-bottom: 1px dashed #667eea;
    }
    
    /* Progress Bar */
    .modern-progress {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
    }
    
    .modern-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
    }
    
    /* Confidence Meter */
    .confidence-meter {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .confidence-bar {
        flex: 1;
        height: 12px;
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Quick Action Buttons */
    .quick-action-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-action-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)


def render_welcome_header(username: str = "Trader"):
    """Render a friendly welcome header"""
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5em; margin-bottom: 8px;">
            👋 Welcome back, {username}!
        </h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">
            Your AI trading assistant is ready to help you make smarter investment decisions
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_start_cards():
    """Render quick start action cards for beginners"""
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="action-card">
            <div class="big-icon">📊</div>
            <h4>View Signals</h4>
            <p style="font-size: 0.9em; opacity: 0.9;">See today's AI recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Signals", key="quick_signals", use_container_width=True):
            st.session_state['quick_nav'] = '🔍 AI Analysis'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="action-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div class="big-icon">🤖</div>
            <h4>Start Auto-Trading</h4>
            <p style="font-size: 0.9em; opacity: 0.9;">Let AI trade for you</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Trading", key="quick_auto", use_container_width=True):
            st.session_state['show_trading_wizard'] = True
            st.session_state['wizard_step'] = 1
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="action-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="big-icon">📈</div>
            <h4>My Portfolio</h4>
            <p style="font-size: 0.9em; opacity: 0.9;">Track your investments</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Portfolio", key="quick_portfolio", use_container_width=True):
            st.session_state['quick_nav'] = '💼 Stock Portfolio'
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="action-card" style="background: linear-gradient(135deg, #f5af19 0%, #f12711 100%);">
            <div class="big-icon">🎓</div>
            <h4>Learn Trading</h4>
            <p style="font-size: 0.9em; opacity: 0.9;">Beginner tutorials</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Learn More", key="quick_learn", use_container_width=True):
            st.session_state['quick_nav'] = '🎓 Learning Hub'
            st.rerun()


def render_signal_card(signal: str, symbol: str, confidence: float, 
                       price: float, explanation: str):
    """Render a beginner-friendly signal card with clear explanation"""
    
    signal_class = f"signal-{signal.lower()}"
    signal_icon = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(signal.upper(), "❓")
    signal_color = {"BUY": "#38ef7d", "SELL": "#f45c43", "HOLD": "#f5576c"}.get(signal.upper(), "#667eea")
    
    st.markdown(f"""
    <div class="{signal_class}">
        <div style="font-size: 36px; margin-bottom: 8px;">{signal_icon}</div>
        <h2 style="margin: 0;">{signal.upper()}</h2>
        <h3 style="margin: 4px 0;">{symbol}</h3>
        <p style="font-size: 1.5em; font-weight: bold;">${price:,.2f}</p>
        
        <div class="confidence-meter" style="margin: 16px 0;">
            <span>AI Confidence:</span>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%; background: white;"></div>
            </div>
            <span style="font-weight: bold;">{confidence:.0f}%</span>
        </div>
        
        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin-top: 12px;">
            <p style="margin: 0; font-size: 0.95em;">
                <strong>💡 What this means:</strong><br>
                {explanation}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_explanation_box(title: str, content: str, box_type: str = "info"):
    """Render an explanation box with beginner-friendly content"""
    
    icon = {"info": "ℹ️", "warning": "⚠️", "success": "✅", "tip": "💡"}.get(box_type, "ℹ️")
    box_class = f"{box_type}-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <strong>{icon} {title}</strong>
        <p style="margin: 8px 0 0 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_progress(current_step: int, total_steps: int, step_labels: List[str]):
    """Render a step progress indicator"""
    
    steps_html = ""
    for i, label in enumerate(step_labels):
        step_num = i + 1
        is_completed = step_num < current_step
        is_active = step_num == current_step
        
        circle_class = "step-circle"
        if is_completed:
            circle_class += " completed"
        elif is_active:
            circle_class += " active"
        
        line_class = "step-line"
        if is_completed:
            line_class += " completed"
        
        steps_html += f"""
        <div class="{circle_class}">{step_num if not is_completed else "✓"}</div>
        """
        if i < len(step_labels) - 1:
            steps_html += f'<div class="{line_class}"></div>'
    
    st.markdown(f"""
    <div class="step-indicator">
        {steps_html}
    </div>
    <div style="display: flex; justify-content: space-between; padding: 0 20px;">
        {"".join(f'<span style="font-size: 0.85em; color: rgba(255,255,255,0.7);">{label}</span>' for label in step_labels)}
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]]):
    """Render a row of metric cards"""
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            value = metric.get('value', 0)
            label = metric.get('label', '')
            icon = metric.get('icon', '')
            color = metric.get('color', '#667eea')
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
                <div class="metric-value" style="color: {color};">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_ai_thinking_box(thinking_text: str):
    """Show users what the AI is analyzing"""
    st.markdown(f"""
    <div class="beginner-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="font-size: 32px;">🧠</div>
            <h3 style="margin: 0;">AI Analysis in Progress</h3>
        </div>
        <p style="color: rgba(255,255,255,0.8);">{thinking_text}</p>
        <div class="modern-progress">
            <div class="modern-progress-fill" style="width: 75%; animation: pulse 1.5s infinite;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_level_selector():
    """Render a visual risk level selector with explanations"""
    
    st.markdown("### 🎯 Choose Your Risk Level")
    st.markdown("*How much risk are you comfortable with?*")
    
    risk_options = {
        "Conservative": {
            "icon": "🛡️",
            "color": "#38ef7d",
            "desc": "Smaller trades, lower risk. Best for beginners or protecting capital.",
            "details": "Targets: 2-5% gains | Max loss: 1-2% per trade"
        },
        "Moderate": {
            "icon": "⚖️", 
            "color": "#667eea",
            "desc": "Balanced approach between growth and safety.",
            "details": "Targets: 5-10% gains | Max loss: 3-5% per trade"
        },
        "Aggressive": {
            "icon": "🚀",
            "color": "#f5a623",
            "desc": "Larger positions for faster growth. More volatility.",
            "details": "Targets: 10-20% gains | Max loss: 5-8% per trade"
        },
        "Extreme": {
            "icon": "🔥",
            "color": "#f45c43",
            "desc": "Maximum growth potential. Only for experienced traders.",
            "details": "Targets: 20%+ gains | Max loss: 10%+ per trade"
        }
    }
    
    cols = st.columns(4)
    selected_risk = st.session_state.get('selected_risk', 'Moderate')
    
    for i, (risk_name, risk_info) in enumerate(risk_options.items()):
        with cols[i]:
            is_selected = selected_risk == risk_name
            border_style = f"3px solid {risk_info['color']}" if is_selected else "1px solid rgba(255,255,255,0.1)"
            
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: {border_style};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                min-height: 180px;
            ">
                <div style="font-size: 36px;">{risk_info['icon']}</div>
                <h4 style="color: {risk_info['color']}; margin: 8px 0;">{risk_name}</h4>
                <p style="font-size: 0.85em; color: rgba(255,255,255,0.7);">{risk_info['desc']}</p>
                <p style="font-size: 0.75em; color: rgba(255,255,255,0.5);">{risk_info['details']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select {risk_name}", key=f"risk_{risk_name}", use_container_width=True):
                st.session_state['selected_risk'] = risk_name
                st.rerun()
    
    return selected_risk


def render_asset_class_selector():
    """Render a visual asset class selector with explanations"""
    
    st.markdown("### 💰 Choose What to Trade")
    st.markdown("*Select the types of assets you want to trade*")
    
    asset_options = {
        "Stocks": {
            "icon": "📊",
            "color": "#667eea",
            "desc": "Trade shares of companies like Apple, Tesla, Amazon",
            "examples": "AAPL, TSLA, NVDA, MSFT",
            "volatility": "Medium",
            "hours": "Mon-Fri 9:30am-4pm ET"
        },
        "Crypto": {
            "icon": "₿",
            "color": "#f7931a",
            "desc": "Trade digital currencies like Bitcoin and Ethereum",
            "examples": "BTC, ETH, SOL, XRP",
            "volatility": "High",
            "hours": "24/7"
        },
        "Forex": {
            "icon": "💱",
            "color": "#11998e",
            "desc": "Trade currency pairs like EUR/USD",
            "examples": "EUR/USD, GBP/USD, USD/JPY",
            "volatility": "Low-Medium",
            "hours": "24/5 (Sun-Fri)"
        },
        "Options": {
            "icon": "📑",
            "color": "#764ba2",
            "desc": "Trade contracts for higher leverage (advanced)",
            "examples": "AAPL Calls, SPY Puts",
            "volatility": "Very High",
            "hours": "Mon-Fri 9:30am-4pm ET"
        }
    }
    
    selected_assets = st.session_state.get('selected_assets', ['Stocks'])
    
    cols = st.columns(4)
    for i, (asset_name, asset_info) in enumerate(asset_options.items()):
        with cols[i]:
            is_selected = asset_name in selected_assets
            border_style = f"3px solid {asset_info['color']}" if is_selected else "1px solid rgba(255,255,255,0.1)"
            check_icon = "✅ " if is_selected else ""
            
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: {border_style};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                min-height: 200px;
            ">
                <div style="font-size: 36px;">{asset_info['icon']}</div>
                <h4 style="color: {asset_info['color']}; margin: 8px 0;">{check_icon}{asset_name}</h4>
                <p style="font-size: 0.85em; color: rgba(255,255,255,0.7);">{asset_info['desc']}</p>
                <p style="font-size: 0.75em; color: rgba(255,255,255,0.5);">
                    <strong>Examples:</strong> {asset_info['examples']}<br>
                    <strong>Volatility:</strong> {asset_info['volatility']}<br>
                    <strong>Hours:</strong> {asset_info['hours']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(
                f"{'Deselect' if is_selected else 'Select'} {asset_name}", 
                key=f"asset_{asset_name}", 
                use_container_width=True
            ):
                if is_selected:
                    selected_assets.remove(asset_name)
                else:
                    selected_assets.append(asset_name)
                st.session_state['selected_assets'] = selected_assets
                st.rerun()
    
    return selected_assets


def render_timeframe_selector():
    """Render a visual timeframe selector with explanations"""
    
    st.markdown("### ⏰ Choose Your Trading Timeframe")
    st.markdown("*How long do you want to hold your trades?*")
    
    timeframe_options = {
        "Scalping": {
            "icon": "⚡",
            "color": "#f45c43",
            "duration": "Minutes to hours",
            "desc": "Very fast trades, requires constant monitoring",
            "frequency": "10-50 trades/day",
            "best_for": "Full-time traders"
        },
        "Day Trading": {
            "icon": "☀️",
            "color": "#f5a623",
            "duration": "Hours (same day)",
            "desc": "Buy and sell within the same day",
            "frequency": "3-10 trades/day",
            "best_for": "Active traders"
        },
        "Swing Trading": {
            "icon": "🌊",
            "color": "#667eea",
            "duration": "Days to weeks",
            "desc": "Hold positions for several days to capture larger moves",
            "frequency": "1-5 trades/week",
            "best_for": "Part-time traders"
        },
        "Position Trading": {
            "icon": "🏔️",
            "color": "#11998e",
            "duration": "Weeks to months",
            "desc": "Longer-term investments for bigger gains",
            "frequency": "1-4 trades/month",
            "best_for": "Long-term investors"
        }
    }
    
    selected_timeframe = st.session_state.get('selected_timeframe', 'Swing Trading')
    
    cols = st.columns(4)
    for i, (tf_name, tf_info) in enumerate(timeframe_options.items()):
        with cols[i]:
            is_selected = selected_timeframe == tf_name
            border_style = f"3px solid {tf_info['color']}" if is_selected else "1px solid rgba(255,255,255,0.1)"
            
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: {border_style};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                min-height: 200px;
            ">
                <div style="font-size: 36px;">{tf_info['icon']}</div>
                <h4 style="color: {tf_info['color']}; margin: 8px 0;">{tf_name}</h4>
                <p style="font-size: 0.9em; font-weight: bold;">{tf_info['duration']}</p>
                <p style="font-size: 0.85em; color: rgba(255,255,255,0.7);">{tf_info['desc']}</p>
                <p style="font-size: 0.75em; color: rgba(255,255,255,0.5);">
                    {tf_info['frequency']}<br>
                    Best for: {tf_info['best_for']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select {tf_name}", key=f"tf_{tf_name}", use_container_width=True):
                st.session_state['selected_timeframe'] = tf_name
                st.rerun()
    
    return selected_timeframe


def render_trading_terms_glossary():
    """Render a beginner-friendly trading terms glossary"""
    
    terms = {
        "📈 Bull Market": "When prices are rising. Investors are optimistic and buying.",
        "📉 Bear Market": "When prices are falling. Investors are pessimistic and selling.",
        "💹 Signal": "A recommendation from the AI about whether to buy, sell, or hold.",
        "🎯 Target Price": "The price where you plan to sell for profit.",
        "🛑 Stop Loss": "A safety limit - automatically sells if price drops too much.",
        "📊 Portfolio": "All the investments you currently own.",
        "💰 Position": "An active trade or investment you currently hold.",
        "⚖️ Risk/Reward": "How much you could lose vs. how much you could gain.",
        "📉 Drawdown": "The drop from highest point to lowest point in your account.",
        "🔄 Volatility": "How much prices move up and down. Higher = more risky.",
    }
    
    st.markdown("### 📚 Trading Terms Explained")
    st.markdown("*Hover or click on any term to learn what it means*")
    
    cols = st.columns(2)
    for i, (term, definition) in enumerate(terms.items()):
        with cols[i % 2]:
            with st.expander(term):
                st.write(definition)


def render_performance_summary_card(metrics: Dict[str, Any]):
    """Render a visual performance summary for beginners"""
    
    total_return = metrics.get('total_return', 0)
    win_rate = metrics.get('win_rate', 0)
    total_trades = metrics.get('total_trades', 0)
    best_trade = metrics.get('best_trade', 0)
    
    return_color = "#38ef7d" if total_return >= 0 else "#f45c43"
    return_icon = "📈" if total_return >= 0 else "📉"
    
    st.markdown(f"""
    <div class="beginner-card">
        <h3 style="margin-bottom: 20px;">📊 Your Performance Summary</h3>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            <div class="metric-card">
                <div style="font-size: 24px;">{return_icon}</div>
                <div class="metric-value" style="color: {return_color};">{total_return:+.1f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            
            <div class="metric-card">
                <div style="font-size: 24px;">🎯</div>
                <div class="metric-value">{win_rate:.0f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            
            <div class="metric-card">
                <div style="font-size: 24px;">📊</div>
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            
            <div class="metric-card">
                <div style="font-size: 24px;">🏆</div>
                <div class="metric-value" style="color: #38ef7d;">+{best_trade:.1f}%</div>
                <div class="metric-label">Best Trade</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
