"""
Twitter/X Sentiment Analysis for IntelliTradeAI
Analyzes social media sentiment for trading signals
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re


class TwitterSentimentAnalyzer:
    """
    Analyzes Twitter/X sentiment for cryptocurrencies and stocks
    Note: For production use, requires Twitter API credentials
    """
    
    def __init__(self):
        self.demo_mode = True
        self.cache = {}
        self.cache_duration = 300
    
    def get_sentiment_score(self, symbol: str, asset_type: str = 'stock') -> Dict[str, Any]:
        """
        Get sentiment score for a symbol
        
        Args:
            symbol: Ticker symbol
            asset_type: 'stock' or 'crypto'
            
        Returns:
            Sentiment analysis with score and metrics
        """
        if self.demo_mode:
            return self._generate_demo_sentiment(symbol, asset_type)
        
        return {
            'symbol': symbol,
            'asset_type': asset_type,
            'sentiment_score': 0,
            'sentiment_label': 'NEUTRAL',
            'message': 'Twitter API integration required for live sentiment analysis',
            'demo_mode': False
        }
    
    def _generate_demo_sentiment(self, symbol: str, asset_type: str) -> Dict[str, Any]:
        """Generate realistic demo sentiment data"""
        import random
        random.seed(hash(symbol + datetime.now().strftime('%Y-%m-%d')))
        
        sentiment_score = random.uniform(-1, 1)
        
        if sentiment_score > 0.5:
            label = 'VERY POSITIVE'
            emoji = '🟢'
        elif sentiment_score > 0.15:
            label = 'POSITIVE'
            emoji = '🟢'
        elif sentiment_score > -0.15:
            label = 'NEUTRAL'
            emoji = '🟡'
        elif sentiment_score > -0.5:
            label = 'NEGATIVE'
            emoji = '🔴'
        else:
            label = 'VERY NEGATIVE'
            emoji = '🔴'
        
        tweet_count = random.randint(500, 5000)
        positive_percent = max(0, min(100, 50 + sentiment_score * 40))
        negative_percent = max(0, min(100, 50 - sentiment_score * 40))
        neutral_percent = 100 - positive_percent - negative_percent
        
        trending_topics = self._generate_trending_topics(symbol, sentiment_score)
        
        return {
            'symbol': symbol,
            'asset_type': asset_type,
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': label,
            'emoji': emoji,
            'confidence': round(abs(sentiment_score) * 100, 1),
            'tweet_volume_24h': tweet_count,
            'sentiment_breakdown': {
                'positive': round(positive_percent, 1),
                'neutral': round(neutral_percent, 1),
                'negative': round(negative_percent, 1)
            },
            'trending_topics': trending_topics,
            'engagement_score': random.randint(60, 95),
            'volume_change_24h': round(random.uniform(-30, 50), 1),
            'demo_mode': True,
            'timestamp': datetime.now().isoformat(),
            'recommendation': self._generate_sentiment_recommendation(sentiment_score, label)
        }
    
    def _generate_trending_topics(self, symbol: str, sentiment: float) -> List[str]:
        """Generate realistic trending topics"""
        import random
        
        positive_topics = [
            f'#{symbol}ToTheMoon', f'#{symbol}Bullish', 'Breaking ATH',
            'Institutional Adoption', 'Major Partnership', 'Innovation'
        ]
        
        negative_topics = [
            'Market Correction', 'Sell-off', 'Regulatory Concerns',
            f'#{symbol}Bearish', 'Profit Taking', 'Market Weakness'
        ]
        
        neutral_topics = [
            f'#{symbol}Analysis', 'Technical Levels', 'Market Watch',
            'Trading Volume', f'{symbol} News', 'Chart Patterns'
        ]
        
        if sentiment > 0.3:
            topics = random.sample(positive_topics, 3) + random.sample(neutral_topics, 1)
        elif sentiment < -0.3:
            topics = random.sample(negative_topics, 3) + random.sample(neutral_topics, 1)
        else:
            topics = random.sample(neutral_topics, 3) + random.sample(positive_topics + negative_topics, 1)
        
        return topics[:4]
    
    def _generate_sentiment_recommendation(self, score: float, label: str) -> str:
        """Generate trading recommendation based on sentiment"""
        if score > 0.5:
            return f"STRONG BUY - {label} sentiment with high conviction. Social momentum is strong."
        elif score > 0.15:
            return f"BUY - {label} sentiment suggests upward momentum. Monitor for confirmation."
        elif score > -0.15:
            return f"HOLD - {label} sentiment. Wait for clearer signals before acting."
        elif score > -0.5:
            return f"CAUTION - {label} sentiment. Consider reducing exposure or wait for reversal."
        else:
            return f"AVOID - {label} sentiment indicates strong bearish mood. High risk."
    
    def get_batch_sentiment(self, symbols: List[str], asset_type: str = 'stock') -> Dict[str, Any]:
        """Get sentiment for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.get_sentiment_score(symbol, asset_type)
        
        return {
            'batch_results': results,
            'total_symbols': len(symbols),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_sentiment_trend(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get sentiment trend over time"""
        import random
        random.seed(hash(symbol))
        
        trend_data = []
        base_sentiment = random.uniform(-0.5, 0.5)
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i-1)).strftime('%Y-%m-%d')
            daily_sentiment = base_sentiment + random.uniform(-0.2, 0.2)
            daily_sentiment = max(-1, min(1, daily_sentiment))
            
            trend_data.append({
                'date': date,
                'sentiment_score': round(daily_sentiment, 3),
                'tweet_volume': random.randint(300, 3000)
            })
        
        avg_sentiment = sum(d['sentiment_score'] for d in trend_data) / len(trend_data)
        
        if trend_data[-1]['sentiment_score'] > trend_data[0]['sentiment_score']:
            trend_direction = 'IMPROVING'
        elif trend_data[-1]['sentiment_score'] < trend_data[0]['sentiment_score']:
            trend_direction = 'DECLINING'
        else:
            trend_direction = 'STABLE'
        
        return {
            'symbol': symbol,
            'period_days': days,
            'trend_data': trend_data,
            'average_sentiment': round(avg_sentiment, 3),
            'trend_direction': trend_direction,
            'current_vs_avg': round((trend_data[-1]['sentiment_score'] - avg_sentiment) * 100, 1),
            'demo_mode': True
        }
