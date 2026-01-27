"""
TradingView-Style Chart Tools for IntelliTradeAI
Provides interactive drawing tools and technical analysis overlays
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class ChartToolbar:
    """TradingView-style toolbar for interactive charting"""
    
    TOOL_CATEGORIES = {
        'drawing': ['trendline', 'horizontal_line', 'vertical_line', 'ray', 'channel'],
        'fibonacci': ['fib_retracement', 'fib_extension', 'fib_fan'],
        'shapes': ['rectangle', 'ellipse', 'triangle', 'arrow'],
        'patterns': ['head_shoulders', 'double_top', 'wedge', 'flag'],
        'indicators': ['sma', 'ema', 'bollinger', 'rsi', 'macd', 'volume']
    }
    
    INDICATOR_COLORS = {
        'sma_20': '#2196F3',
        'sma_50': '#FF9800',
        'sma_200': '#9C27B0',
        'ema_12': '#00BCD4',
        'ema_26': '#E91E63',
        'bollinger_upper': '#4CAF50',
        'bollinger_lower': '#4CAF50',
        'bollinger_middle': '#FFC107'
    }
    
    def __init__(self, chart_id: str):
        self.chart_id = chart_id
        self.active_tool = None
        self.drawings = []
        self.indicators = []
    
    @staticmethod
    def render_toolbar(chart_key: str) -> Dict[str, Any]:
        """Render the chart toolbar with all tools"""
        
        st.markdown("""
        <style>
        .chart-toolbar {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #0f3460;
        }
        .tool-btn {
            background: #0f3460;
            border: none;
            color: white;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 4px;
            cursor: pointer;
        }
        .tool-btn:hover {
            background: #e94560;
        }
        </style>
        """, unsafe_allow_html=True)
        
        toolbar_config = {}
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("**📐 Drawing**")
            toolbar_config['show_trendline'] = st.checkbox("Auto Trendline", key=f"{chart_key}_trendline")
            toolbar_config['show_hline'] = st.checkbox("H-Line", key=f"{chart_key}_hline")
            toolbar_config['show_vline'] = st.checkbox("V-Line", key=f"{chart_key}_vline")
        
        with col2:
            st.markdown("**📊 Fibonacci**")
            toolbar_config['show_fib_retracement'] = st.checkbox("Retracement", key=f"{chart_key}_fib_ret")
            toolbar_config['show_fib_extension'] = st.checkbox("Extension", key=f"{chart_key}_fib_ext")
        
        with col3:
            st.markdown("**📈 Moving Averages**")
            toolbar_config['show_sma_20'] = st.checkbox("SMA 20", key=f"{chart_key}_sma20", value=True)
            toolbar_config['show_sma_50'] = st.checkbox("SMA 50", key=f"{chart_key}_sma50")
            toolbar_config['show_sma_200'] = st.checkbox("SMA 200", key=f"{chart_key}_sma200")
        
        with col4:
            st.markdown("**📉 EMA**")
            toolbar_config['show_ema_12'] = st.checkbox("EMA 12", key=f"{chart_key}_ema12")
            toolbar_config['show_ema_26'] = st.checkbox("EMA 26", key=f"{chart_key}_ema26")
        
        with col5:
            st.markdown("**🔧 Overlays**")
            toolbar_config['show_bollinger'] = st.checkbox("Bollinger Bands", key=f"{chart_key}_boll")
            toolbar_config['show_volume'] = st.checkbox("Volume", key=f"{chart_key}_vol", value=True)
        
        return toolbar_config
    
    @staticmethod
    def render_indicator_panel(chart_key: str) -> Dict[str, Any]:
        """Render additional indicator selection panel"""
        
        with st.expander("🔬 Advanced Indicators", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Momentum**")
                show_rsi = st.checkbox("RSI (14)", key=f"{chart_key}_rsi")
                show_stoch = st.checkbox("Stochastic", key=f"{chart_key}_stoch")
                show_cci = st.checkbox("CCI", key=f"{chart_key}_cci")
            
            with col2:
                st.markdown("**Trend**")
                show_macd = st.checkbox("MACD", key=f"{chart_key}_macd")
                show_adx = st.checkbox("ADX", key=f"{chart_key}_adx")
                show_parabolic = st.checkbox("Parabolic SAR", key=f"{chart_key}_psar")
            
            with col3:
                st.markdown("**Volatility**")
                show_atr = st.checkbox("ATR", key=f"{chart_key}_atr")
                show_keltner = st.checkbox("Keltner Channel", key=f"{chart_key}_keltner")
                show_donchian = st.checkbox("Donchian Channel", key=f"{chart_key}_donchian")
        
        return {
            'show_rsi': show_rsi,
            'show_stoch': show_stoch,
            'show_cci': show_cci,
            'show_macd': show_macd,
            'show_adx': show_adx,
            'show_parabolic': show_parabolic,
            'show_atr': show_atr,
            'show_keltner': show_keltner,
            'show_donchian': show_donchian
        }
    
    @staticmethod
    def render_custom_trendline_controls(chart_key: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Render controls for drawing custom trendlines"""
        
        custom_lines = []
        
        # Initialize session state for custom trendlines
        if f"{chart_key}_custom_lines" not in st.session_state:
            st.session_state[f"{chart_key}_custom_lines"] = []
        
        with st.expander("✏️ Draw Custom Trendline", expanded=True):
            st.markdown("**Set your trendline start and end points:**")
            
            # Get price range for sliders
            min_price = float(df['low'].min()) * 0.95
            max_price = float(df['high'].max()) * 1.05
            current_price = float(df['close'].iloc[-1])
            
            # Get date range
            dates = df.index.tolist()
            num_bars = len(dates)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Start Point**")
                start_bar = st.slider(
                    "Start Bar (from left)", 
                    min_value=0, 
                    max_value=num_bars-1, 
                    value=max(0, num_bars - 50),
                    key=f"{chart_key}_start_bar"
                )
                start_price = st.number_input(
                    "Start Price ($)", 
                    min_value=min_price,
                    max_value=max_price,
                    value=float(df['low'].iloc[start_bar]) if start_bar < len(df) else min_price,
                    step=0.01,
                    key=f"{chart_key}_start_price"
                )
            
            with col2:
                st.markdown("**End Point**")
                end_bar = st.slider(
                    "End Bar (from left)", 
                    min_value=0, 
                    max_value=num_bars-1, 
                    value=num_bars-1,
                    key=f"{chart_key}_end_bar"
                )
                end_price = st.number_input(
                    "End Price ($)", 
                    min_value=min_price,
                    max_value=max_price,
                    value=float(df['low'].iloc[end_bar]) if end_bar < len(df) else current_price,
                    step=0.01,
                    key=f"{chart_key}_end_price"
                )
            
            col3, col4 = st.columns(2)
            with col3:
                line_color = st.color_picker("Line Color", "#FFD700", key=f"{chart_key}_line_color")
            with col4:
                line_style = st.selectbox("Line Style", ["Solid", "Dashed", "Dotted"], key=f"{chart_key}_line_style")
            
            col5, col6 = st.columns(2)
            with col5:
                if st.button("➕ Add Trendline", key=f"{chart_key}_add_line", use_container_width=True):
                    new_line = {
                        'start_date': dates[start_bar],
                        'start_price': start_price,
                        'end_date': dates[end_bar],
                        'end_price': end_price,
                        'color': line_color,
                        'style': line_style
                    }
                    st.session_state[f"{chart_key}_custom_lines"].append(new_line)
                    st.rerun()
            
            with col6:
                if st.button("🗑️ Clear All Lines", key=f"{chart_key}_clear_lines", use_container_width=True):
                    st.session_state[f"{chart_key}_custom_lines"] = []
                    st.rerun()
            
            # Show existing lines
            if st.session_state[f"{chart_key}_custom_lines"]:
                st.markdown("**Your Trendlines:**")
                for i, line in enumerate(st.session_state[f"{chart_key}_custom_lines"]):
                    direction = "↗️ Up" if line['end_price'] > line['start_price'] else "↘️ Down"
                    st.caption(f"{i+1}. {direction}: ${line['start_price']:.2f} → ${line['end_price']:.2f}")
        
        return {
            'custom_lines': st.session_state.get(f"{chart_key}_custom_lines", [])
        }


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    df = df.copy()
    
    if 'close' not in df.columns:
        return df
    
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['atr'] = calculate_atr(df, period=14)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 100):
    """Calculate Fibonacci retracement levels"""
    recent_data = df.tail(lookback)
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    diff = high - low
    
    return {
        '0.0%': high,
        '23.6%': high - (diff * 0.236),
        '38.2%': high - (diff * 0.382),
        '50.0%': high - (diff * 0.5),
        '61.8%': high - (diff * 0.618),
        '78.6%': high - (diff * 0.786),
        '100.0%': low
    }


def create_advanced_chart(
    df: pd.DataFrame,
    symbol: str,
    toolbar_config: Dict[str, Any],
    indicator_config: Dict[str, Any] = None,
    chart_key: str = "default"
):
    """Create advanced chart with TradingView-style tools and indicators"""
    
    df = calculate_indicators(df)
    
    num_rows = 1
    row_heights = [0.7]
    
    if indicator_config:
        if indicator_config.get('show_rsi'):
            num_rows += 1
            row_heights.append(0.15)
        if indicator_config.get('show_macd'):
            num_rows += 1
            row_heights.append(0.15)
    
    if toolbar_config.get('show_volume'):
        num_rows += 1
        row_heights.append(0.15)
    
    total_height = sum(row_heights)
    row_heights = [h / total_height for h in row_heights]
    
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # TradingView-style candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#26A69A',
            increasing_fillcolor='#26A69A',
            decreasing_line_color='#EF5350',
            decreasing_fillcolor='#EF5350',
            line=dict(width=1)
        ),
        row=1, col=1
    )
    
    # Add current price annotation on right side (TradingView style)
    current_price = df['close'].iloc[-1]
    fig.add_annotation(
        x=df.index[-1],
        y=current_price,
        text=f"${current_price:,.2f}",
        showarrow=True,
        arrowhead=0,
        arrowcolor='#2962FF',
        ax=40,
        ay=0,
        font=dict(size=11, color='white'),
        bgcolor='#2962FF',
        bordercolor='#2962FF',
        borderwidth=1,
        borderpad=4,
        xanchor='left'
    )
    
    if toolbar_config.get('show_sma_20') and 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20',
                      line=dict(color='#2196F3', width=1)),
            row=1, col=1
        )
    
    if toolbar_config.get('show_sma_50') and 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50',
                      line=dict(color='#FF9800', width=1)),
            row=1, col=1
        )
    
    if toolbar_config.get('show_sma_200') and 'sma_200' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_200'], name='SMA 200',
                      line=dict(color='#9C27B0', width=1.5)),
            row=1, col=1
        )
    
    if toolbar_config.get('show_ema_12') and 'ema_12' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_12'], name='EMA 12',
                      line=dict(color='#00BCD4', width=1, dash='dot')),
            row=1, col=1
        )
    
    if toolbar_config.get('show_ema_26') and 'ema_26' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_26'], name='EMA 26',
                      line=dict(color='#E91E63', width=1, dash='dot')),
            row=1, col=1
        )
    
    if toolbar_config.get('show_bollinger'):
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                          line=dict(color='#4CAF50', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                          line=dict(color='#4CAF50', width=1, dash='dash'),
                          fill='tonexty', fillcolor='rgba(76, 175, 80, 0.1)'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_middle'], name='BB Middle',
                          line=dict(color='#FFC107', width=1)),
                row=1, col=1
            )
    
    # Add trendline if enabled - draws from swing lows (support line)
    if toolbar_config.get('show_trendline'):
        lookback = min(100, len(df))
        recent_df = df.tail(lookback).copy()
        dates = recent_df.index.tolist()
        
        # Find swing lows (local minima)
        lows = recent_df['low'].values
        swing_low_indices = []
        
        # Find local minima with window of 5
        window = 5
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_low_indices.append(i)
        
        # Need at least 2 swing lows for a trendline
        if len(swing_low_indices) >= 2:
            # Use the first and last swing lows for the trendline
            x_indices = np.array(swing_low_indices)
            y_lows = lows[swing_low_indices]
            
            # Linear regression on swing lows
            slope, intercept = np.polyfit(x_indices, y_lows, 1)
            
            # Calculate trendline from first swing low to end of chart
            first_idx = swing_low_indices[0]
            last_idx = len(recent_df) - 1
            
            trend_x_range = np.arange(first_idx, last_idx + 1)
            trend_y = slope * trend_x_range + intercept
            
            # Get corresponding dates
            trend_dates = [dates[i] for i in range(first_idx, last_idx + 1)]
            
            # Determine trend direction
            trend_direction = "Uptrend" if slope > 0 else "Downtrend"
            trend_color = "#16C784" if slope > 0 else "#EA3943"
            
            fig.add_trace(
                go.Scatter(
                    x=trend_dates,
                    y=trend_y,
                    name=f"Support Line ({trend_direction})",
                    line=dict(color=trend_color, width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Mark swing low points
            swing_low_dates = [dates[i] for i in swing_low_indices]
            swing_low_prices = lows[swing_low_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=swing_low_dates,
                    y=swing_low_prices,
                    name="Swing Lows",
                    mode='markers',
                    marker=dict(color=trend_color, size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
    
    # Add custom trendlines if enabled
    if toolbar_config.get('show_custom_trendline'):
        custom_lines = toolbar_config.get('custom_lines', [])
        for i, line in enumerate(custom_lines):
            dash_style = 'solid'
            if line.get('style') == 'Dashed':
                dash_style = 'dash'
            elif line.get('style') == 'Dotted':
                dash_style = 'dot'
            
            fig.add_trace(
                go.Scatter(
                    x=[line['start_date'], line['end_date']],
                    y=[line['start_price'], line['end_price']],
                    name=f"Trendline {i+1}",
                    mode='lines',
                    line=dict(color=line.get('color', '#FFD700'), width=2, dash=dash_style)
                ),
                row=1, col=1
            )
            
            # Add markers at start and end points
            fig.add_trace(
                go.Scatter(
                    x=[line['start_date'], line['end_date']],
                    y=[line['start_price'], line['end_price']],
                    mode='markers',
                    marker=dict(color=line.get('color', '#FFD700'), size=8, symbol='circle'),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add horizontal line if enabled
    if toolbar_config.get('show_hline'):
        # Draw horizontal line at current price
        current_price = df['close'].iloc[-1]
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="#FFD700",
            line_width=2,
            annotation_text=f"Current: ${current_price:,.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Add vertical line if enabled (at latest date)
    if toolbar_config.get('show_vline'):
        fig.add_vline(
            x=df.index[-1],
            line_dash="solid",
            line_color="#00BFFF",
            line_width=1,
            annotation_text="Now",
            annotation_position="top",
            row=1, col=1
        )
    
    if toolbar_config.get('show_fib_retracement'):
        fib_levels = calculate_fibonacci_levels(df)
        colors = ['#F44336', '#FF9800', '#FFEB3B', '#4CAF50', '#2196F3', '#9C27B0', '#795548']
        for i, (level, price) in enumerate(fib_levels.items()):
            fig.add_hline(
                y=price, 
                line_dash="dash", 
                line_color=colors[i % len(colors)],
                annotation_text=f"Fib {level}: ${price:,.2f}",
                annotation_position="right",
                row=1, col=1
            )
    
    current_row = 2
    
    if toolbar_config.get('show_volume') and 'volume' in df.columns:
        # TradingView-style volume colors (teal/red matching candles)
        colors = ['#26A69A' if df['close'].iloc[i] >= df['open'].iloc[i] else '#EF5350' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume',
                   marker_color=colors, opacity=0.5),
            row=current_row, col=1
        )
        fig.update_yaxes(
            title_text="Volume", 
            row=current_row, col=1,
            gridcolor='#363c4e',
            tickfont=dict(color='#787b86', size=10),
            title_font=dict(color='#787b86', size=11)
        )
        current_row += 1
    
    if indicator_config and indicator_config.get('show_rsi') and 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                      line=dict(color='#9C27B0', width=1.5)),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    if indicator_config and indicator_config.get('show_macd'):
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD',
                          line=dict(color='#2196F3', width=1.5)),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                          line=dict(color='#FF9800', width=1)),
                row=current_row, col=1
            )
            colors = ['#16C784' if val >= 0 else '#EA3943' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['macd_hist'], name='MACD Hist',
                       marker_color=colors, opacity=0.5),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    # TradingView-style dark theme
    fig.update_layout(
        title=dict(
            text=f"{symbol} - Advanced Technical Analysis",
            font=dict(size=16, color='#d1d4dc'),
            x=0.01
        ),
        height=650 + (num_rows - 1) * 120,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color='#d1d4dc'),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=80, t=60, b=40),
        hovermode='x unified'
    )
    
    # Style axes like TradingView
    fig.update_yaxes(
        title_text="Price ($)", 
        row=1, col=1,
        gridcolor='#363c4e',
        gridwidth=0.5,
        tickfont=dict(color='#787b86', size=10),
        title_font=dict(color='#787b86', size=11),
        side='right',
        showgrid=True
    )
    fig.update_xaxes(
        title_text="", 
        row=num_rows, col=1,
        gridcolor='#363c4e',
        gridwidth=0.5,
        tickfont=dict(color='#787b86', size=10),
        showgrid=True
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#1a1a2e",
            activecolor="#e94560"
        ),
        row=1, col=1
    )
    
    # Enable editable shapes for mouse-drawn trendlines
    fig.update_layout(
        dragmode='drawline',  # Set default mode to draw lines
        newshape=dict(
            line=dict(color='#FFD700', width=2),
            fillcolor='rgba(255, 215, 0, 0.3)'
        ),
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='white',
            activecolor='#e94560'
        )
    )
    
    config = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath', 
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': ['autoScale2d', 'lasso2d', 'select2d'],
        'displayModeBar': True,
        'displaylogo': False,
        'editable': True,  # Allow editing of shapes
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{symbol}_chart',
            'height': 800,
            'width': 1400,
            'scale': 2
        }
    }
    
    return fig, config


def render_chart_with_toolbar(
    df: pd.DataFrame,
    symbol: str,
    chart_key: str = "chart"
) -> None:
    """Render a complete chart with toolbar"""
    
    st.markdown(f"### 📊 {symbol} Technical Analysis")
    
    # Drawing instructions
    st.info("✏️ **Draw on Chart:** Click and drag directly on the chart to draw trendlines. Use the toolbar icons at top-right: 📏 Line | ✏️ Freehand | ⭕ Circle | ⬜ Rectangle | 🗑️ Erase")
    
    toolbar_config = ChartToolbar.render_toolbar(chart_key)
    indicator_config = ChartToolbar.render_indicator_panel(chart_key)
    
    if df is not None and len(df) > 0:
        fig, config = create_advanced_chart(
            df, symbol, toolbar_config, indicator_config, chart_key
        )
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = df['close'].iloc[-1] if 'close' in df.columns else 0
            st.metric("Current Price", f"${current_price:,.2f}")
        with col2:
            if len(df) > 1:
                change = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
                st.metric("24h Change", f"{change:+.2f}%")
        with col3:
            if 'high' in df.columns:
                high_52w = df['high'].tail(252).max() if len(df) >= 252 else df['high'].max()
                st.metric("52W High", f"${high_52w:,.2f}")
        with col4:
            if 'low' in df.columns:
                low_52w = df['low'].tail(252).min() if len(df) >= 252 else df['low'].min()
                st.metric("52W Low", f"${low_52w:,.2f}")
    else:
        st.warning("No data available for charting")
