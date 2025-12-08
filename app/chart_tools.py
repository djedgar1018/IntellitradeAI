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
            toolbar_config['show_trendline'] = st.checkbox("Trendline", key=f"{chart_key}_trendline")
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
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#16C784',
            decreasing_line_color='#EA3943'
        ),
        row=1, col=1
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
        recent_df = recent_df.reset_index(drop=False)
        
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
            trend_dates = recent_df.iloc[first_idx:last_idx + 1]['index'].values
            
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
            swing_low_dates = recent_df.iloc[swing_low_indices]['index'].values
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
        colors = ['#16C784' if df['close'].iloc[i] >= df['open'].iloc[i] else '#EA3943' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
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
    
    fig.update_layout(
        title=f"{symbol} - Advanced Technical Analysis",
        height=600 + (num_rows - 1) * 100,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
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
    
    config = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': ['autoScale2d'],
        'displayModeBar': True,
        'displaylogo': False,
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
