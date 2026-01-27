"""
UI Components for IntelliTradeAI Dashboard
Reusable visualization components including Fear & Greed Speedometer
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import numpy as np


def create_fear_greed_speedometer(value: int, title: str = "Fear & Greed Index", 
                                   size: int = 300, classification: str = None,
                                   color: str = None) -> go.Figure:
    """
    Create a speedometer-style gauge for Fear & Greed Index
    Similar to CoinMarketCap.com style
    
    Args:
        value: Index value 0-100
        title: Chart title
        size: Size of the gauge
        classification: Optional classification from data source
        color: Optional color from data source
        
    Returns:
        Plotly figure with speedometer gauge
    """
    
    if classification is None:
        if value <= 25:
            classification = "Extreme Fear"
        elif value <= 45:
            classification = "Fear"
        elif value <= 55:
            classification = "Neutral"
        elif value <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"
    
    if color is None:
        if value <= 25:
            bar_color = "#EA3943"
        elif value <= 45:
            bar_color = "#F5A623"
        elif value <= 55:
            bar_color = "#F5D033"
        elif value <= 75:
            bar_color = "#93D900"
        else:
            bar_color = "#16C784"
    else:
        bar_color = color
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "", 'font': {'size': 48, 'color': bar_color, 'family': 'Arial Black'}},
        title={'text': f"<b>{title}</b><br><span style='font-size:16px;color:{bar_color}'>{classification}</span>",
               'font': {'size': 18}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#666",
                'tickmode': 'array',
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['0', '25', '50', '75', '100'],
                'tickfont': {'size': 12}
            },
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': "rgba(234, 57, 67, 0.3)"},
                {'range': [25, 45], 'color': "rgba(245, 166, 35, 0.3)"},
                {'range': [45, 55], 'color': "rgba(245, 208, 51, 0.3)"},
                {'range': [55, 75], 'color': "rgba(147, 217, 0, 0.3)"},
                {'range': [75, 100], 'color': "rgba(22, 199, 132, 0.3)"}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=size,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    
    return fig


def create_overall_sentiment_gauge(value: float, title: str = "Overall Market Sentiment",
                                    classification: str = None, color: str = None) -> go.Figure:
    """Create a larger overall sentiment speedometer"""
    return create_fear_greed_speedometer(int(value), title, size=350, 
                                          classification=classification, color=color)


def render_fear_greed_legend():
    """Render the Fear & Greed index legend"""
    st.markdown("""
    <style>
    .fng-legend {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #EA3943 0%, #F5A623 25%, #F5D033 50%, #93D900 75%, #16C784 100%);
        margin-top: 10px;
    }
    .fng-legend-item {
        text-align: center;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-size: 11px;
    }
    </style>
    <div class="fng-legend">
        <div class="fng-legend-item">Extreme<br>Fear<br>0-25</div>
        <div class="fng-legend-item">Fear<br>25-45</div>
        <div class="fng-legend-item">Neutral<br>45-55</div>
        <div class="fng-legend-item">Greed<br>55-75</div>
        <div class="fng-legend-item">Extreme<br>Greed<br>75-100</div>
    </div>
    """, unsafe_allow_html=True)


def render_asset_trading_mode_toggle(asset_type: str, mode_manager) -> str:
    """
    Render trading mode toggle for specific asset type
    
    Args:
        asset_type: 'crypto', 'stocks', or 'options'
        mode_manager: TradingModeManager instance
        
    Returns:
        Current mode for this asset type
    """
    from trading.mode_manager import TradingMode
    
    mode_key = f"{asset_type}_trading_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "manual"
    
    current_mode = st.session_state[mode_key]
    
    col1, col2 = st.columns(2)
    
    with col1:
        manual_selected = current_mode == "manual"
        if st.button(
            f"👤 Manual",
            key=f"manual_{asset_type}",
            type="primary" if manual_selected else "secondary",
            use_container_width=True,
            help="AI provides recommendations, you make the final decision"
        ):
            st.session_state[mode_key] = "manual"
            st.rerun()
    
    with col2:
        auto_selected = current_mode == "automatic"
        if st.button(
            f"🤖 Automatic",
            key=f"auto_{asset_type}",
            type="primary" if auto_selected else "secondary",
            use_container_width=True,
            help="AI executes trades autonomously based on signals"
        ):
            st.session_state[mode_key] = "automatic"
            st.rerun()
    
    mode_icon = "👤" if current_mode == "manual" else "🤖"
    mode_text = "Manual (AI-Assisted)" if current_mode == "manual" else "Automatic (AI Executes)"
    
    if current_mode == "manual":
        st.info(f"{mode_icon} **{mode_text}**: AI provides recommendations as assists. You make the final trading decision.")
    else:
        st.warning(f"{mode_icon} **{mode_text}**: AI will automatically execute trades when confidence thresholds are met.")
    
    return current_mode


def render_all_trading_mode_toggles():
    """Render trading mode toggles for all asset classes and sync with TradingModeManager"""
    from trading.mode_manager import TradingModeManager, TradingMode
    
    if 'mode_manager' not in st.session_state:
        st.session_state['mode_manager'] = TradingModeManager(TradingMode.MANUAL)
    
    mode_manager = st.session_state['mode_manager']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🪙 Crypto")
        crypto_mode = render_single_toggle("crypto", mode_manager)
    
    with col2:
        st.markdown("### 📈 Stocks")
        stocks_mode = render_single_toggle("stocks", mode_manager)
    
    with col3:
        st.markdown("### 📊 Options")
        options_mode = render_single_toggle("options", mode_manager)
    
    return {
        'crypto': crypto_mode,
        'stocks': stocks_mode,
        'options': options_mode
    }


def render_single_toggle(asset_type: str, mode_manager=None) -> str:
    """Render a single toggle switch for asset type and update mode manager using public API"""
    mode_key = f"{asset_type}_trading_mode"
    
    if mode_manager:
        current_mode = mode_manager.get_asset_mode(asset_type)
        st.session_state[mode_key] = current_mode
    elif mode_key not in st.session_state:
        st.session_state[mode_key] = "manual"
    
    current_mode = st.session_state[mode_key]
    is_auto = current_mode == "automatic"
    
    toggle = st.toggle(
        f"🤖 Automatic Mode",
        value=is_auto,
        key=f"toggle_{asset_type}",
        help=f"Enable automatic AI trading for {asset_type}"
    )
    
    new_mode = "automatic" if toggle else "manual"
    
    if new_mode != current_mode:
        st.session_state[mode_key] = new_mode
        
        if mode_manager:
            result = mode_manager.set_asset_mode(asset_type, new_mode)
            if result.get('success'):
                st.toast(f"✅ {asset_type.title()} mode changed to {new_mode}", icon="🔄")
    
    if toggle:
        st.caption("🤖 AI will execute trades automatically")
    else:
        st.caption("🚗 AI provides recommendations only")
    
    return st.session_state[mode_key]


def create_options_price_projection_chart(
    current_price: float,
    strike_price: float,
    option_type: str,
    expiry_days: int = 30,
    projected_move: float = 0.1,
    confidence: float = 75.0
) -> go.Figure:
    """
    Create a price projection chart for options recommendation
    
    Args:
        current_price: Current stock price
        strike_price: Option strike price
        option_type: 'CALL' or 'PUT'
        expiry_days: Days until expiration
        projected_move: Expected price movement (as decimal, e.g., 0.1 = 10%)
        confidence: AI confidence in this projection
        
    Returns:
        Plotly figure with price projection
    """
    import pandas as pd
    
    days = list(range(expiry_days + 1))
    
    if option_type == 'CALL':
        target_price = current_price * (1 + projected_move)
        upper_bound = current_price * (1 + projected_move * 1.5)
        lower_bound = current_price * (1 - projected_move * 0.3)
    else:
        target_price = current_price * (1 - projected_move)
        upper_bound = current_price * (1 + projected_move * 0.3)
        lower_bound = current_price * (1 - projected_move * 1.5)
    
    np.random.seed(42)
    
    base_path = np.linspace(current_price, target_price, expiry_days + 1)
    noise = np.random.normal(0, current_price * 0.01, expiry_days + 1)
    main_path = base_path + noise * np.linspace(0, 1, expiry_days + 1)
    
    upper_path = np.linspace(current_price, upper_bound, expiry_days + 1)
    lower_path = np.linspace(current_price, lower_bound, expiry_days + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days,
        y=upper_path,
        mode='lines',
        line=dict(color='rgba(0,200,0,0.3)', width=0),
        name='Upper Bound',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=lower_path,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(100,149,237,0.2)',
        line=dict(color='rgba(0,100,200,0.3)', width=0),
        name='Probability Range',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=main_path,
        mode='lines',
        line=dict(color='#1E90FF', width=3),
        name='Projected Price Path'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, expiry_days],
        y=[strike_price, strike_price],
        mode='lines',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        name=f'Strike: ${strike_price:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, expiry_days],
        y=[current_price, current_price],
        mode='lines',
        line=dict(color='#888', width=1, dash='dot'),
        name=f'Current: ${current_price:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[expiry_days],
        y=[target_price],
        mode='markers+text',
        marker=dict(size=12, color='#16C784', symbol='star'),
        text=[f'Target: ${target_price:.2f}'],
        textposition='top center',
        name='Target Price'
    ))
    
    profit_zone_color = '#16C784' if option_type == 'CALL' else '#EA3943'
    
    fig.update_layout(
        title=dict(
            text=f"📈 {option_type} Option Price Projection<br><sub>Confidence: {confidence:.1f}%</sub>",
            font=dict(size=16)
        ),
        xaxis_title="Days Until Expiration",
        yaxis_title="Stock Price ($)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    return fig


def render_options_recommendation_popup(
    symbol: str,
    option_type: str,
    strike: float,
    premium: float,
    current_price: float,
    delta: float = 0.5,
    recommendation: Optional[Dict[str, Any]] = None
):
    """
    Render a popup/modal with options recommendation details
    including price projection chart and probability analysis
    """
    expiry_days = 30
    
    if option_type == 'CALL':
        projected_move = 0.08
        success_prob = min(85, 50 + delta * 50)
    else:
        projected_move = -0.08
        success_prob = min(85, 50 + abs(delta) * 50)
    
    confidence = recommendation.get('confidence', 72) if recommendation else 72
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; border: 2px solid #0f3460; margin: 10px 0;">
        <h3 style="color: #e94560; margin: 0;">📊 {symbol} {option_type} Option Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strike Price", f"${strike:,.2f}")
    with col2:
        st.metric("Premium", f"${premium:,.2f}")
    with col3:
        st.metric("Current Price", f"${current_price:,.2f}")
    with col4:
        breakeven = strike + premium if option_type == 'CALL' else strike - premium
        st.metric("Breakeven", f"${breakeven:,.2f}")
    
    st.markdown("---")
    
    fig = create_options_price_projection_chart(
        current_price=current_price,
        strike_price=strike,
        option_type=option_type,
        expiry_days=expiry_days,
        projected_move=abs(projected_move),
        confidence=confidence
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Probability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        profit_color = "green" if success_prob > 60 else "orange" if success_prob > 40 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: linear-gradient(180deg, rgba(22,199,132,0.1) 0%, rgba(22,199,132,0.05) 100%); border-radius: 10px;">
            <div style="font-size: 36px; font-weight: bold; color: {profit_color};">{success_prob:.1f}%</div>
            <div style="color: #888;">Probability of Profit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        loss_prob = 100 - success_prob
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: linear-gradient(180deg, rgba(234,57,67,0.1) 0%, rgba(234,57,67,0.05) 100%); border-radius: 10px;">
            <div style="font-size: 36px; font-weight: bold; color: #EA3943;">{loss_prob:.1f}%</div>
            <div style="color: #888;">Probability of Loss</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_loss = premium * 100
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: linear-gradient(180deg, rgba(100,149,237,0.1) 0%, rgba(100,149,237,0.05) 100%); border-radius: 10px;">
            <div style="font-size: 36px; font-weight: bold; color: #6495ED;">${max_loss:.0f}</div>
            <div style="color: #888;">Max Loss (per contract)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 AI Confidence Assessment")
    
    confidence_color = "#16C784" if confidence >= 70 else "#F5A623" if confidence >= 50 else "#EA3943"
    
    st.progress(confidence / 100)
    st.markdown(f"""
    <div style="text-align: center; margin-top: -10px;">
        <span style="font-size: 24px; font-weight: bold; color: {confidence_color};">{confidence:.1f}% Confidence</span>
    </div>
    """, unsafe_allow_html=True)
    
    if confidence >= 70:
        st.success("✅ High confidence trade - AI strongly recommends this position")
    elif confidence >= 50:
        st.warning("⚠️ Moderate confidence - Consider smaller position size")
    else:
        st.error("❌ Low confidence - High risk trade, proceed with caution")
    
    if recommendation and 'reason' in recommendation:
        st.info(f"💡 **AI Analysis:** {recommendation['reason']}")


def render_goal_based_optimizer():
    """
    Render the goal-based trading optimizer UI.
    Allows users to set their trading goals and get optimized parameters.
    """
    st.markdown("### 🎯 Goal-Based Strategy Optimizer")
    st.markdown("Tell us your trading goals and we'll optimize parameters to help you achieve them.")
    
    try:
        from trading.goal_based_optimizer import GoalBasedOptimizer
        optimizer = GoalBasedOptimizer()
    except ImportError:
        st.error("Goal-based optimizer module not available")
        return
    
    goal_col1, goal_col2 = st.columns(2)
    
    with goal_col1:
        target_multiple = st.selectbox(
            "Target Growth",
            options=[1.5, 2.0, 3.0, 5.0, 10.0, 20.0],
            format_func=lambda x: f"{x}x ({(x-1)*100:.0f}% return)",
            index=2,
            help="Your target portfolio growth"
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            options=[7, 14, 30, 60, 90],
            format_func=lambda x: f"{x} days" if x < 30 else f"{x//30} month{'s' if x > 30 else ''}",
            index=2,
            help="Time to achieve your goal"
        )
        
        starting_capital = st.number_input(
            "Starting Capital ($)",
            min_value=100.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0
        )
    
    with goal_col2:
        asset_class = st.selectbox(
            "Asset Class",
            options=['all', 'stocks', 'crypto', 'forex', 'options'],
            format_func=lambda x: x.title() if x != 'all' else 'All Assets',
            index=0,
            help="Focus on specific asset class or trade all"
        )
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=['conservative', 'moderate', 'aggressive', 'extreme'],
            format_func=lambda x: x.title(),
            index=2,
            help="How much risk are you willing to take?"
        )
    
    if st.button("🚀 Generate Optimized Strategy", type="primary", use_container_width=True):
        with st.spinner("Calculating optimal parameters for your goal..."):
            recommendation = optimizer.get_recommendation(
                target_multiple=target_multiple,
                timeframe_days=timeframe,
                asset_class=asset_class,
                risk_tolerance=risk_tolerance,
                starting_capital=starting_capital
            )
            
            st.session_state['trading_goal_recommendation'] = recommendation
    
    if 'trading_goal_recommendation' in st.session_state:
        rec = st.session_state['trading_goal_recommendation']
        
        st.markdown("---")
        st.markdown("### 📊 Your Optimized Trading Plan")
        
        goal_info = rec.get('user_goal', {})
        feasibility = rec.get('feasibility_assessment', {})
        
        score = feasibility.get('score', 0)
        if score >= 80:
            score_color = "green"
            score_emoji = "✅"
        elif score >= 60:
            score_color = "orange"
            score_emoji = "⚠️"
        elif score >= 40:
            score_color = "red"
            score_emoji = "⚡"
        else:
            score_color = "red"
            score_emoji = "🔥"
        
        cols = st.columns(4)
        cols[0].metric("Target", goal_info.get('target_multiple', 'N/A'))
        cols[1].metric("Target Balance", f"${goal_info.get('target_balance', 0):,.0f}")
        cols[2].metric("Timeframe", f"{goal_info.get('timeframe_days', 0)} days")
        cols[3].metric(
            f"{score_emoji} Feasibility",
            f"{score:.0f}%",
            delta=feasibility.get('rating', 'Unknown')
        )
        
        if feasibility.get('message'):
            st.info(f"💡 {feasibility['message']}")
        
        perf_req = rec.get('performance_requirements', {})
        st.markdown("#### 📈 Performance Requirements")
        perf_cols = st.columns(4)
        perf_cols[0].metric("Daily Return Needed", perf_req.get('daily_return_needed', 'N/A'))
        perf_cols[1].metric("Weekly Return Needed", perf_req.get('weekly_return_needed', 'N/A'))
        perf_cols[2].metric("Required Win Rate", perf_req.get('required_win_rate', 'N/A'))
        perf_cols[3].metric("Avg Gain/Trade", perf_req.get('required_avg_gain_per_trade', 'N/A'))
        
        params = rec.get('optimized_parameters', {})
        
        with st.expander("🔧 Optimized Parameters", expanded=True):
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                st.markdown("**Position Sizing**")
                sizing = params.get('position_sizing', {})
                st.write(f"Base Risk: {sizing.get('base_risk_pct', 0)}%")
                st.write(f"Max Position: {sizing.get('max_position_pct', 0)}%")
                st.write(f"Max Positions: {sizing.get('max_positions', 0)}")
            
            with param_col2:
                st.markdown("**Exit Rules**")
                exits = params.get('exit_rules', {})
                st.write(f"Stop Loss: {exits.get('stop_loss_pct', 0)}%")
                st.write(f"Target: {exits.get('target_pct', 0)}%")
                st.write(f"Max Hold: {exits.get('max_hold_days', 0)} days")
            
            with param_col3:
                st.markdown("**Compounding**")
                comp = params.get('compounding', {})
                st.write(f"Pyramiding: {'Enabled' if comp.get('pyramid_enabled') else 'Disabled'}")
                st.write(f"Max Adds: {comp.get('pyramid_max', 0)}x")
                st.write(f"Win Streak Bonus: {comp.get('win_streak_mult_max', 0)}x")
        
        recommendations = rec.get('recommendations', [])
        if recommendations:
            with st.expander("💡 Recommendations", expanded=True):
                for i, r in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {r}")
        
        apply_col1, apply_col2 = st.columns(2)
        
        with apply_col1:
            if st.button("✅ Apply These Parameters", type="primary", use_container_width=True):
                try:
                    from trading.mode_manager import TradingModeManager
                    manager = TradingModeManager()
                    if manager.apply_goal_parameters(rec):
                        st.success("Parameters applied successfully! Your trading is now optimized for your goal.")
                        st.balloons()
                    else:
                        st.error("Failed to apply parameters")
                except Exception as e:
                    st.error(f"Error applying parameters: {e}")
        
        with apply_col2:
            if st.button("📊 Compare Asset Classes", use_container_width=True):
                comparison = optimizer.compare_asset_classes(target_multiple, timeframe)
                st.session_state['asset_comparison'] = comparison
        
        if 'asset_comparison' in st.session_state:
            comp = st.session_state['asset_comparison']
            st.markdown("#### 🏆 Asset Class Comparison")
            st.caption(f"Goal: {comp.get('goal', 'N/A')}")
            
            comp_data = comp.get('comparison', {})
            if comp_data:
                import pandas as pd
                df = pd.DataFrame([
                    {
                        'Asset': asset.title(),
                        'Feasibility': f"{data.get('feasibility_score', 0):.0f}%",
                        'Rating': data.get('rating', 'N/A'),
                        'Win Rate Needed': data.get('required_win_rate', 'N/A'),
                        'Stop Loss': f"{data.get('optimized_stop', 0)}%",
                        'Target': f"{data.get('optimized_target', 0)}%"
                    }
                    for asset, data in comp_data.items()
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                best = comp.get('best_choice', 'crypto')
                st.success(f"🏆 **Recommended Asset Class:** {best.title()} - Best feasibility for your goal")
