"""
Social Sentiment Analysis Integration
Aggregates sentiment from multiple sources for trading signals
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os


class SocialSentimentAnalyzer:
    """Analyzes social media sentiment for crypto assets"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 1800
    
    def get_aggregate_sentiment(self, symbol: str) -> Dict:
        """
        Get aggregated sentiment score from multiple sources
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'DOGE')
            
        Returns:
            Dictionary with sentiment metrics
        """
        sentiment = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        sentiment['sources']['fear_greed'] = self._get_fear_greed_index()
        sentiment['sources']['price_sentiment'] = self._estimate_price_sentiment(symbol)
        sentiment['sources']['volume_sentiment'] = self._estimate_volume_sentiment(symbol)
        sentiment['sources']['social_buzz'] = self._estimate_social_buzz(symbol)
        
        sentiment['aggregate_score'] = self._calculate_aggregate_score(sentiment['sources'])
        sentiment['signal'] = self._sentiment_to_signal(sentiment['aggregate_score'])
        
        return sentiment
    
    def _get_fear_greed_index(self) -> Dict:
        """Fetch Fear & Greed Index from Alternative.me API"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    fng = data['data'][0]
                    return {
                        'value': int(fng['value']),
                        'classification': fng['value_classification'],
                        'source': 'alternative.me',
                        'status': 'success'
                    }
            
            return self._default_fear_greed()
            
        except Exception as e:
            return self._default_fear_greed()
    
    def _default_fear_greed(self) -> Dict:
        return {
            'value': 50,
            'classification': 'Neutral',
            'source': 'default',
            'status': 'unavailable'
        }
    
    def _estimate_price_sentiment(self, symbol: str) -> Dict:
        """Estimate sentiment based on recent price action"""
        try:
            import yfinance as yf
            
            yf_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period='7d', interval='1h')
            
            if hist.empty:
                return {'score': 50, 'trend': 'neutral', 'status': 'unavailable'}
            
            returns_1d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-24]) / hist['Close'].iloc[-24] * 100 if len(hist) >= 24 else 0
            returns_7d = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            
            if returns_1d > 5:
                score = min(90, 50 + returns_1d * 4)
                trend = 'very_bullish'
            elif returns_1d > 2:
                score = min(75, 50 + returns_1d * 5)
                trend = 'bullish'
            elif returns_1d < -5:
                score = max(10, 50 + returns_1d * 4)
                trend = 'very_bearish'
            elif returns_1d < -2:
                score = max(25, 50 + returns_1d * 5)
                trend = 'bearish'
            else:
                score = 50 + returns_1d * 3
                trend = 'neutral'
            
            return {
                'score': round(score, 1),
                'trend': trend,
                'returns_1d': round(returns_1d, 2),
                'returns_7d': round(returns_7d, 2),
                'status': 'success'
            }
            
        except Exception as e:
            return {'score': 50, 'trend': 'neutral', 'status': 'error'}
    
    def _estimate_volume_sentiment(self, symbol: str) -> Dict:
        """Estimate sentiment based on volume patterns"""
        try:
            import yfinance as yf
            
            yf_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period='30d', interval='1d')
            
            if hist.empty or len(hist) < 7:
                return {'score': 50, 'volume_trend': 'neutral', 'status': 'unavailable'}
            
            avg_vol = hist['Volume'].mean()
            recent_vol = hist['Volume'].tail(3).mean()
            volume_ratio = recent_vol / (avg_vol + 1e-10)
            
            recent_price_change = hist['Close'].pct_change().tail(3).mean()
            
            if volume_ratio > 1.5 and recent_price_change > 0:
                score = 75
                trend = 'accumulation'
            elif volume_ratio > 1.5 and recent_price_change < 0:
                score = 25
                trend = 'distribution'
            elif volume_ratio > 1.2:
                score = 60 if recent_price_change > 0 else 40
                trend = 'elevated'
            else:
                score = 50
                trend = 'normal'
            
            return {
                'score': round(score, 1),
                'volume_trend': trend,
                'volume_ratio': round(volume_ratio, 2),
                'status': 'success'
            }
            
        except Exception as e:
            return {'score': 50, 'volume_trend': 'neutral', 'status': 'error'}
    
    def _estimate_social_buzz(self, symbol: str) -> Dict:
        """
        Estimate social buzz level based on asset characteristics
        (Proxy when direct social data unavailable)
        """
        high_buzz_assets = ['DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'FLOKI', 
                           'MEME', 'BRETT', 'POPCAT', 'GOAT', 'AI16Z', 'VIRTUAL']
        
        medium_buzz_assets = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 
                              'LINK', 'DOT', 'MATIC', 'FET', 'RNDR', 'APE']
        
        if symbol in high_buzz_assets:
            base_buzz = 80
            volatility_factor = 1.3
        elif symbol in medium_buzz_assets:
            base_buzz = 60
            volatility_factor = 1.0
        else:
            base_buzz = 40
            volatility_factor = 0.8
        
        noise = np.random.uniform(-5, 5)
        
        return {
            'buzz_score': round(min(100, max(0, base_buzz + noise)), 1),
            'buzz_level': 'high' if base_buzz >= 70 else 'medium' if base_buzz >= 50 else 'low',
            'volatility_factor': volatility_factor,
            'status': 'estimated'
        }
    
    def _calculate_aggregate_score(self, sources: Dict) -> float:
        """Calculate weighted aggregate sentiment score (0-100)"""
        weights = {
            'fear_greed': 0.25,
            'price_sentiment': 0.35,
            'volume_sentiment': 0.25,
            'social_buzz': 0.15
        }
        
        scores = {
            'fear_greed': sources.get('fear_greed', {}).get('value', 50),
            'price_sentiment': sources.get('price_sentiment', {}).get('score', 50),
            'volume_sentiment': sources.get('volume_sentiment', {}).get('score', 50),
            'social_buzz': sources.get('social_buzz', {}).get('buzz_score', 50)
        }
        
        aggregate = sum(scores[k] * weights[k] for k in weights)
        
        return round(aggregate, 1)
    
    def _sentiment_to_signal(self, score: float) -> str:
        """Convert sentiment score to trading signal"""
        if score >= 80:
            return 'STRONG_BUY'
        elif score >= 65:
            return 'BUY'
        elif score >= 55:
            return 'WEAK_BUY'
        elif score >= 45:
            return 'NEUTRAL'
        elif score >= 35:
            return 'WEAK_SELL'
        elif score >= 20:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def calculate_sentiment_features(self, symbol: str) -> Dict:
        """
        Calculate sentiment-derived features for ML model
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dictionary with numerical features
        """
        sentiment = self.get_aggregate_sentiment(symbol)
        
        features = {
            'sentiment_fear_greed': sentiment['sources'].get('fear_greed', {}).get('value', 50) / 100,
            'sentiment_price': sentiment['sources'].get('price_sentiment', {}).get('score', 50) / 100,
            'sentiment_volume': sentiment['sources'].get('volume_sentiment', {}).get('score', 50) / 100,
            'sentiment_social_buzz': sentiment['sources'].get('social_buzz', {}).get('buzz_score', 50) / 100,
            'sentiment_aggregate': sentiment.get('aggregate_score', 50) / 100
        }
        
        signal_map = {
            'STRONG_BUY': 1.0,
            'BUY': 0.75,
            'WEAK_BUY': 0.6,
            'NEUTRAL': 0.5,
            'WEAK_SELL': 0.4,
            'SELL': 0.25,
            'STRONG_SELL': 0.0
        }
        features['sentiment_signal_score'] = signal_map.get(sentiment.get('signal', 'NEUTRAL'), 0.5)
        
        return features


class SentimentFeatureEngineer:
    """Integrates sentiment into feature engineering pipeline"""
    
    def __init__(self):
        self.sentiment_analyzer = SocialSentimentAnalyzer()
    
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add sentiment-derived features to a DataFrame
        
        Args:
            df: DataFrame with price data
            symbol: Crypto symbol
            
        Returns:
            DataFrame with additional sentiment features
        """
        features = self.sentiment_analyzer.calculate_sentiment_features(symbol)
        
        for feat_name, feat_value in features.items():
            df[feat_name] = feat_value
        
        return df
