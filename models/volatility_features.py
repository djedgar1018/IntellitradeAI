"""
Volatility-Aware Feature Engineering for High-Volatility Assets
Implements specialized features for meme coins, AI agents, and NFT tokens
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class VolatilityFeatures:
    """Feature engineering specifically designed for volatile crypto assets"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive volatility-aware features
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with additional volatility features
        """
        df = data.copy()
        
        df = self._calculate_realized_volatility(df)
        df = self._calculate_volatility_regime(df)
        df = self._calculate_drawdown_features(df)
        df = self._calculate_liquidity_features(df)
        df = self._calculate_momentum_intensity(df)
        df = self._calculate_extreme_move_features(df)
        df = self._calculate_mean_reversion_signals(df)
        df = self._calculate_volatility_clustering(df)
        
        return df
    
    def _calculate_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility at multiple timeframes"""
        returns = df['close'].pct_change()
        
        for window in [5, 10, 20, 30]:
            df[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        df['vol_ratio_5_20'] = df['realized_vol_5d'] / (df['realized_vol_20d'] + 1e-10)
        df['vol_ratio_10_30'] = df['realized_vol_10d'] / (df['realized_vol_30d'] + 1e-10)
        
        df['vol_percentile_20d'] = returns.abs().rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        return df
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility regime (low/normal/high/extreme)"""
        vol_20 = df['realized_vol_20d']
        vol_mean = vol_20.rolling(60).mean()
        vol_std = vol_20.rolling(60).std()
        
        z_score = (vol_20 - vol_mean) / (vol_std + 1e-10)
        
        df['vol_z_score'] = z_score
        
        df['vol_regime'] = 0
        df.loc[z_score < -1, 'vol_regime'] = -1
        df.loc[(z_score >= -1) & (z_score < 1), 'vol_regime'] = 0
        df.loc[(z_score >= 1) & (z_score < 2), 'vol_regime'] = 1
        df.loc[z_score >= 2, 'vol_regime'] = 2
        
        return df
    
    def _calculate_drawdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown-related features"""
        prices = df['close']
        
        rolling_max = prices.rolling(window=20, min_periods=1).max()
        drawdown = (prices - rolling_max) / rolling_max
        df['drawdown_20d'] = drawdown
        
        rolling_max_60 = prices.rolling(window=60, min_periods=1).max()
        drawdown_60 = (prices - rolling_max_60) / rolling_max_60
        df['drawdown_60d'] = drawdown_60
        
        df['max_drawdown_20d'] = drawdown.rolling(20).min()
        
        df['drawdown_recovery'] = np.where(
            df['drawdown_20d'] < -0.05,
            (df['drawdown_20d'] - df['drawdown_20d'].shift(1)) > 0,
            0
        ).astype(float)
        
        days_since_high = np.zeros(len(prices))
        last_high_idx = 0
        for i in range(len(prices)):
            if prices.iloc[i] >= rolling_max.iloc[i] * 0.99:
                last_high_idx = i
            days_since_high[i] = i - last_high_idx
        df['days_since_high'] = days_since_high
        
        return df
    
    def _calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity proxy features"""
        df['dollar_volume'] = df['close'] * df['volume']
        
        for window in [5, 10, 20]:
            df[f'avg_dollar_volume_{window}d'] = df['dollar_volume'].rolling(window).mean()
        
        df['volume_ratio_5_20'] = (
            df['avg_dollar_volume_5d'] / (df['avg_dollar_volume_20d'] + 1e-10)
        )
        
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['avg_spread_10d'] = df['hl_spread'].rolling(10).mean()
        
        amihud = df['close'].pct_change().abs() / (df['dollar_volume'] + 1e-10)
        df['amihud_illiquidity'] = amihud.rolling(10).mean()
        
        return df
    
    def _calculate_momentum_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum intensity features for volatile assets"""
        returns = df['close'].pct_change()
        
        for window in [3, 5, 10]:
            df[f'momentum_{window}d'] = df['close'].pct_change(window)
        
        df['momentum_intensity'] = returns.abs().rolling(5).sum()
        
        df['up_intensity_5d'] = returns.where(returns > 0, 0).rolling(5).sum()
        df['down_intensity_5d'] = returns.where(returns < 0, 0).abs().rolling(5).sum()
        df['intensity_ratio'] = df['up_intensity_5d'] / (df['down_intensity_5d'] + 1e-10)
        
        sign_returns = pd.Series(np.sign(returns.values), index=returns.index)
        df['consecutive_direction'] = sign_returns.rolling(5).apply(
            lambda x: np.abs(x.sum()) / len(x) if len(x) > 0 else 0
        )
        
        return df
    
    def _calculate_extreme_move_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and quantify extreme price movements"""
        returns = df['close'].pct_change()
        
        ret_mean = returns.rolling(30).mean()
        ret_std = returns.rolling(30).std()
        
        df['return_z_score'] = (returns - ret_mean) / (ret_std + 1e-10)
        
        df['extreme_up_count_20d'] = (returns > ret_mean + 2 * ret_std).rolling(20).sum()
        df['extreme_down_count_20d'] = (returns < ret_mean - 2 * ret_std).rolling(20).sum()
        
        df['gap_up'] = (df['open'] / df['close'].shift(1) - 1).clip(lower=0)
        df['gap_down'] = (1 - df['open'] / df['close'].shift(1)).clip(lower=0)
        df['avg_gap_5d'] = (df['gap_up'] + df['gap_down']).rolling(5).mean()
        
        intraday_range = (df['high'] - df['low']) / df['close']
        avg_range = intraday_range.rolling(20).mean()
        df['range_expansion'] = intraday_range / (avg_range + 1e-10)
        
        return df
    
    def _calculate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators for volatile assets"""
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            df[f'dist_from_sma_{window}'] = (df['close'] - sma) / sma
        
        window = 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        df['rsi_14'] = rsi
        
        df['rsi_extreme'] = 0
        df.loc[rsi < 30, 'rsi_extreme'] = -1
        df.loc[rsi > 70, 'rsi_extreme'] = 1
        
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-10)
        
        return df
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility clustering patterns"""
        returns = df['close'].pct_change()
        squared_returns = returns ** 2
        
        df['garch_proxy'] = squared_returns.ewm(span=10).mean()
        
        df['vol_autocorr'] = squared_returns.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else 0
        )
        
        high_vol_days = squared_returns > squared_returns.rolling(30).quantile(0.9)
        df['high_vol_streak'] = high_vol_days.groupby(
            (~high_vol_days).cumsum()
        ).cumsum()
        
        df['vol_surprise'] = (
            squared_returns / (squared_returns.rolling(10).mean() + 1e-10)
        )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of all volatility feature names"""
        return [
            'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d', 'realized_vol_30d',
            'vol_ratio_5_20', 'vol_ratio_10_30', 'vol_percentile_20d',
            'vol_z_score', 'vol_regime',
            'drawdown_20d', 'drawdown_60d', 'max_drawdown_20d', 
            'drawdown_recovery', 'days_since_high',
            'dollar_volume', 'avg_dollar_volume_5d', 'avg_dollar_volume_10d', 
            'avg_dollar_volume_20d', 'volume_ratio_5_20', 
            'hl_spread', 'avg_spread_10d', 'amihud_illiquidity',
            'momentum_3d', 'momentum_5d', 'momentum_10d',
            'momentum_intensity', 'up_intensity_5d', 'down_intensity_5d',
            'intensity_ratio', 'consecutive_direction',
            'return_z_score', 'extreme_up_count_20d', 'extreme_down_count_20d',
            'gap_up', 'gap_down', 'avg_gap_5d', 'range_expansion',
            'dist_from_sma_10', 'dist_from_sma_20', 'dist_from_sma_50',
            'rsi_14', 'rsi_extreme', 'bb_position',
            'garch_proxy', 'vol_autocorr', 'high_vol_streak', 'vol_surprise'
        ]


class AdaptiveThresholdEngine:
    """Engine for determining optimal prediction thresholds based on volatility"""
    
    VOLATILITY_THRESHOLDS = {
        'extreme': {
            'min_move_pct': 10.0,
            'horizons': [3, 5, 7],
            'class_balance_range': (0.05, 0.95)
        },
        'very_high': {
            'min_move_pct': 7.0,
            'horizons': [5, 7, 10],
            'class_balance_range': (0.06, 0.94)
        },
        'high': {
            'min_move_pct': 5.0,
            'horizons': [5, 7, 10],
            'class_balance_range': (0.07, 0.93)
        },
        'standard': {
            'min_move_pct': 4.0,
            'horizons': [5, 7, 10],
            'class_balance_range': (0.08, 0.92)
        }
    }
    
    @classmethod
    def get_optimal_thresholds(cls, symbol: str, historical_volatility: float = None) -> Dict:
        """
        Get optimal training thresholds for a given symbol
        
        Args:
            symbol: Trading symbol
            historical_volatility: Optional pre-computed volatility
            
        Returns:
            Dictionary with optimal thresholds and horizons
        """
        try:
            from config.assets_config import CryptoAssets
            volatility_class = CryptoAssets.get_volatility_class(symbol)
        except:
            volatility_class = 'standard'
        
        config = cls.VOLATILITY_THRESHOLDS.get(volatility_class, cls.VOLATILITY_THRESHOLDS['standard'])
        
        base_threshold = config['min_move_pct']
        
        if volatility_class == 'extreme':
            thresholds = [base_threshold, base_threshold * 1.25, base_threshold * 1.5, base_threshold * 2.0]
        elif volatility_class == 'very_high':
            thresholds = [base_threshold, base_threshold * 1.2, base_threshold * 1.5]
        else:
            thresholds = [base_threshold, base_threshold * 1.25, base_threshold * 1.5]
        
        return {
            'volatility_class': volatility_class,
            'thresholds': thresholds,
            'horizons': config['horizons'],
            'class_balance_range': config['class_balance_range']
        }
    
    @classmethod
    def calculate_dynamic_threshold(cls, data: pd.DataFrame, base_threshold: float = 5.0) -> float:
        """
        Calculate dynamic threshold based on recent price volatility
        
        Args:
            data: DataFrame with price data
            base_threshold: Base threshold percentage
            
        Returns:
            Adjusted threshold
        """
        if len(data) < 30:
            return base_threshold
        
        returns = data['close'].pct_change().dropna()
        recent_vol = returns.tail(30).std() * np.sqrt(252) * 100
        
        vol_ratio = recent_vol / 50.0
        
        adjusted_threshold = base_threshold * np.clip(vol_ratio, 0.5, 3.0)
        
        return round(adjusted_threshold, 1)
