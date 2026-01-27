"""
Automated Trading Setup Wizard for IntelliTradeAI
Step-by-step guide to start automated trading for beginners
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

try:
    from app.beginner_friendly_ui import (
        render_step_progress, render_explanation_box,
        render_risk_level_selector, render_asset_class_selector,
        render_timeframe_selector, render_metric_row
    )
except ImportError:
    from beginner_friendly_ui import (
        render_step_progress, render_explanation_box,
        render_risk_level_selector, render_asset_class_selector,
        render_timeframe_selector, render_metric_row
    )


class AutomatedTradingWizard:
    """Step-by-step wizard for setting up automated trading"""
    
    STEPS = [
        "Welcome",
        "Risk Profile", 
        "Assets",
        "Timeframe",
        "Capital",
        "Review",
        "Activate"
    ]
    
    def __init__(self):
        if 'wizard_step' not in st.session_state:
            st.session_state['wizard_step'] = 1
        if 'wizard_config' not in st.session_state:
            st.session_state['wizard_config'] = {}
    
    def render(self):
        """Render the full wizard"""
        
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1>🤖 Automated Trading Setup</h1>
            <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">
                Let's set up your AI trading assistant in just a few steps
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        current_step = st.session_state.get('wizard_step', 1)
        render_step_progress(current_step, len(self.STEPS), self.STEPS)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if current_step == 1:
            self._render_welcome_step()
        elif current_step == 2:
            self._render_risk_step()
        elif current_step == 3:
            self._render_assets_step()
        elif current_step == 4:
            self._render_timeframe_step()
        elif current_step == 5:
            self._render_capital_step()
        elif current_step == 6:
            self._render_review_step()
        elif current_step == 7:
            self._render_activation_step()
    
    def _render_navigation(self, show_back: bool = True, show_next: bool = True, 
                          next_label: str = "Continue →"):
        """Render navigation buttons"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if show_back and st.session_state.get('wizard_step', 1) > 1:
                if st.button("← Back", use_container_width=True):
                    st.session_state['wizard_step'] -= 1
                    st.rerun()
        
        with col3:
            if show_next:
                if st.button(next_label, type="primary", use_container_width=True):
                    st.session_state['wizard_step'] += 1
                    st.rerun()
    
    def _render_welcome_step(self):
        """Step 1: Welcome and introduction"""
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin: 20px 0;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">🤖</div>
            <h2 style="margin: 0 0 16px 0;">Welcome to Automated Trading!</h2>
            <p style="font-size: 1.1em; opacity: 0.9; max-width: 600px; margin: 0 auto;">
                In just a few minutes, you'll have an AI assistant managing your trades 24/7.
                No experience needed - we'll guide you through everything.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 What you'll set up:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin: 8px 0;">
                <h4>🎯 Your Risk Profile</h4>
                <p style="color: rgba(255,255,255,0.7);">Tell us how much risk you're comfortable with</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin: 8px 0;">
                <h4>💰 What to Trade</h4>
                <p style="color: rgba(255,255,255,0.7);">Choose stocks, crypto, forex, or options</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin: 8px 0;">
                <h4>⏰ Trading Speed</h4>
                <p style="color: rgba(255,255,255,0.7);">How often should the AI make trades?</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin: 8px 0;">
                <h4>💵 Your Capital</h4>
                <p style="color: rgba(255,255,255,0.7);">How much are you starting with?</p>
            </div>
            """, unsafe_allow_html=True)
        
        render_explanation_box(
            "Paper Trading Mode",
            "Don't worry! You'll start in Paper Trading mode - this means the AI practices with fake money first. "
            "You can see how it performs before risking real money.",
            "info"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        self._render_navigation(show_back=False, next_label="Let's Get Started →")
    
    def _render_risk_step(self):
        """Step 2: Risk profile selection"""
        
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h3>🎯 Understanding Risk</h3>
            <p style="color: rgba(255,255,255,0.7);">
                <strong>Risk = Potential Reward</strong><br>
                Higher risk means the AI will take bigger positions, which can lead to bigger gains 
                OR bigger losses. Lower risk means smaller, safer trades.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_risk = render_risk_level_selector()
        st.session_state['wizard_config']['risk_level'] = selected_risk
        
        risk_explanations = {
            "Conservative": "Perfect for beginners! The AI will make careful, smaller trades to protect your capital.",
            "Moderate": "A balanced approach - good potential returns with reasonable protection.",
            "Aggressive": "The AI will take larger positions for faster growth. More exciting, but also more volatile.",
            "Extreme": "Maximum growth mode. Only choose this if you can handle significant ups and downs."
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_explanation_box(
            f"You selected: {selected_risk}",
            risk_explanations.get(selected_risk, ""),
            "success"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        self._render_navigation()
    
    def _render_assets_step(self):
        """Step 3: Asset class selection"""
        
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h3>💡 Which assets should you trade?</h3>
            <p style="color: rgba(255,255,255,0.7);">
                <strong>For beginners:</strong> Start with Stocks - they're the most stable and easiest to understand.<br>
                <strong>For more action:</strong> Add Crypto - trades 24/7 with more price movement.<br>
                <strong>Advanced traders:</strong> Options provide leverage but higher risk.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_assets = render_asset_class_selector()
        st.session_state['wizard_config']['asset_classes'] = selected_assets
        
        if not selected_assets:
            render_explanation_box(
                "Select at least one asset type",
                "Click on the asset cards above to choose what you want to trade.",
                "warning"
            )
        else:
            asset_text = ", ".join(selected_assets)
            render_explanation_box(
                f"You're trading: {asset_text}",
                f"The AI will scan {len(selected_assets)} market{'s' if len(selected_assets) > 1 else ''} for the best opportunities.",
                "success"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        can_continue = len(selected_assets) > 0
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state['wizard_step'] -= 1
                st.rerun()
        with col3:
            if st.button("Continue →", type="primary", use_container_width=True, disabled=not can_continue):
                st.session_state['wizard_step'] += 1
                st.rerun()
    
    def _render_timeframe_step(self):
        """Step 4: Timeframe selection"""
        
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h3>⏱️ How patient are you?</h3>
            <p style="color: rgba(255,255,255,0.7);">
                <strong>Quick trades:</strong> More action, more fees, needs more attention.<br>
                <strong>Longer trades:</strong> Fewer trades, bigger moves, more hands-off.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_timeframe = render_timeframe_selector()
        st.session_state['wizard_config']['timeframe'] = selected_timeframe
        
        timeframe_advice = {
            "Scalping": "⚠️ Scalping requires fast execution. Make sure you have a stable internet connection.",
            "Day Trading": "The AI will enter and exit all positions within the same trading day.",
            "Swing Trading": "Recommended for most users! A good balance of activity and results.",
            "Position Trading": "The AI will hold positions for weeks, capturing larger market moves."
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_explanation_box(
            f"You selected: {selected_timeframe}",
            timeframe_advice.get(selected_timeframe, ""),
            "success" if selected_timeframe != "Scalping" else "warning"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        self._render_navigation()
    
    def _render_capital_step(self):
        """Step 5: Starting capital"""
        
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h3>💵 How much are you investing?</h3>
            <p style="color: rgba(255,255,255,0.7);">
                This helps the AI calculate proper position sizes. 
                <strong>Only invest money you can afford to lose.</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            starting_capital = st.number_input(
                "Starting Capital ($)",
                min_value=100,
                max_value=10000000,
                value=st.session_state['wizard_config'].get('starting_capital', 10000),
                step=1000,
                help="Enter the amount you want to start trading with"
            )
            st.session_state['wizard_config']['starting_capital'] = starting_capital
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Your Starting Capital", f"${starting_capital:,.0f}")
        
        risk_level = st.session_state['wizard_config'].get('risk_level', 'Moderate')
        risk_pct = {"Conservative": 0.01, "Moderate": 0.02, "Aggressive": 0.05, "Extreme": 0.10}
        max_risk_per_trade = starting_capital * risk_pct.get(risk_level, 0.02)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 What this means for your trading:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.2); border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 24px;">💰</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #667eea;">${max_risk_per_trade:,.0f}</div>
                <div style="font-size: 0.9em; color: rgba(255,255,255,0.7);">Max risk per trade</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            positions = {"Conservative": 5, "Moderate": 8, "Aggressive": 10, "Extreme": 15}
            max_positions = positions.get(risk_level, 8)
            st.markdown(f"""
            <div style="background: rgba(17, 153, 142, 0.2); border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 24px;">📊</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #11998e;">{max_positions}</div>
                <div style="font-size: 0.9em; color: rgba(255,255,255,0.7);">Max simultaneous trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            per_position = starting_capital / max_positions
            st.markdown(f"""
            <div style="background: rgba(245, 166, 35, 0.2); border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 24px;">💵</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #f5a623;">${per_position:,.0f}</div>
                <div style="font-size: 0.9em; color: rgba(255,255,255,0.7);">Per trade (approx)</div>
            </div>
            """, unsafe_allow_html=True)
        
        render_explanation_box(
            "Paper Trading First",
            "Remember: You'll start in Paper Trading mode with simulated money. "
            "When you're ready to trade for real, you can switch to Live mode.",
            "info"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        self._render_navigation()
    
    def _render_review_step(self):
        """Step 6: Review all settings"""
        
        config = st.session_state.get('wizard_config', {})
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        ">
            <div style="font-size: 48px; margin-bottom: 12px;">✅</div>
            <h2 style="margin: 0;">Almost Done!</h2>
            <p style="opacity: 0.9;">Review your trading setup before activating</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Your Trading Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0;">
                <h4 style="margin: 0 0 12px 0;">🎯 Risk Level</h4>
                <p style="font-size: 1.3em; font-weight: bold; color: #667eea; margin: 0;">
                    {config.get('risk_level', 'Moderate')}
                </p>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0;">
                <h4 style="margin: 0 0 12px 0;">⏰ Trading Style</h4>
                <p style="font-size: 1.3em; font-weight: bold; color: #667eea; margin: 0;">
                    {config.get('timeframe', 'Swing Trading')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            assets = config.get('asset_classes', ['Stocks'])
            assets_text = ", ".join(assets) if assets else "Not selected"
            capital = config.get('starting_capital', 10000)
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0;">
                <h4 style="margin: 0 0 12px 0;">💰 Asset Classes</h4>
                <p style="font-size: 1.3em; font-weight: bold; color: #667eea; margin: 0;">
                    {assets_text}
                </p>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0;">
                <h4 style="margin: 0 0 12px 0;">💵 Starting Capital</h4>
                <p style="font-size: 1.3em; font-weight: bold; color: #667eea; margin: 0;">
                    ${capital:,}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_explanation_box(
            "What happens next?",
            "After activation, the AI will start scanning markets and making paper trades. "
            "You can monitor performance in real-time and adjust settings anytime.",
            "info"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        self._render_navigation(next_label="Activate Trading →")
    
    def _render_activation_step(self):
        """Step 7: Activation confirmation"""
        
        config = st.session_state.get('wizard_config', {})
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin-bottom: 24px;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">🎉</div>
            <h2 style="margin: 0 0 12px 0;">Your AI Trading Assistant is Ready!</h2>
            <p style="opacity: 0.9; font-size: 1.1em;">
                Paper Trading mode is now active
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 What the AI is doing:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 36px; margin-bottom: 12px;">🔍</div>
                <h4>Scanning Markets</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                    Analyzing thousands of data points to find opportunities
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 36px; margin-bottom: 12px;">📊</div>
                <h4>Generating Signals</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                    Finding buy and sell signals based on your settings
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 36px; margin-bottom: 12px;">📈</div>
                <h4>Executing Trades</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                    Automatically placing paper trades when signals appear
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📱 Quick Tips for Monitoring")
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                <div>
                    <strong>📊 Dashboard</strong><br>
                    <span style="color: rgba(255,255,255,0.7);">View real-time signals and market analysis</span>
                </div>
                <div>
                    <strong>💼 Portfolio</strong><br>
                    <span style="color: rgba(255,255,255,0.7);">Track all your open positions</span>
                </div>
                <div>
                    <strong>📈 Performance</strong><br>
                    <span style="color: rgba(255,255,255,0.7);">See how the AI is performing over time</span>
                </div>
                <div>
                    <strong>⚙️ Settings</strong><br>
                    <span style="color: rgba(255,255,255,0.7);">Adjust risk and trading parameters</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Modify Settings", use_container_width=True):
                st.session_state['wizard_step'] = 2
                st.rerun()
        
        with col2:
            if st.button("📊 View Dashboard", type="primary", use_container_width=True):
                st.session_state['show_trading_wizard'] = False
                st.session_state['wizard_complete'] = True
                st.session_state['auto_trading_active'] = True
                st.rerun()
        
        with col3:
            if st.button("📈 View Signals", use_container_width=True):
                st.session_state['show_trading_wizard'] = False
                st.session_state['wizard_complete'] = True
                st.session_state['auto_trading_active'] = True
                st.session_state['nav_page'] = 'signals'
                st.rerun()


def render_trading_wizard():
    """Main entry point to render the trading wizard"""
    wizard = AutomatedTradingWizard()
    wizard.render()
