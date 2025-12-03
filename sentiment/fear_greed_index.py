"""
Fear & Greed Index for IntelliTradeAI
Provides market sentiment indicators for crypto, stocks, and options
"""

import requests
from typing import Dict, Optional, Any
from datetime import datetime
import random


class FearGreedIndexAnalyzer:
    """
    Analyzes Fear & Greed Index for different asset classes
    Provides market sentiment indicators
    """
    
    def __init__(self):
        self.crypto_api_url = 'https://api.alternative.me/fng/'
        self.demo_mode = True
        self.cache = {}
        self.cache_duration = 3600
    
    def get_crypto_fear_greed(self) -> Dict[str, Any]:
        """Get crypto Fear & Greed Index"""
        try:
            if not self.demo_mode:
                response = requests.get(self.crypto_api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        fng = data['data'][0]
                        value = int(fng['value'])
                        return self._format_fear_greed_response(value, 'crypto', fng.get('value_classification'))
            
            return self._generate_demo_crypto_fear_greed()
            
        except Exception as e:
            return self._generate_demo_crypto_fear_greed()
    
    def get_stock_fear_greed(self) -> Dict[str, Any]:
        """Get stock market Fear & Greed Index"""
        return self._generate_demo_stock_fear_greed()
    
    def get_options_fear_greed(self) -> Dict[str, Any]:
        """Get options market sentiment indicator"""
        return self._generate_demo_options_fear_greed()
    
    def get_all_indices(self) -> Dict[str, Any]:
        """Get Fear & Greed indices for all asset classes"""
        return {
            'crypto': self.get_crypto_fear_greed(),
            'stocks': self.get_stock_fear_greed(),
            'options': self.get_options_fear_greed(),
            'overall_market_sentiment': self._calculate_overall_sentiment(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_demo_crypto_fear_greed(self) -> Dict[str, Any]:
        """Generate demo crypto Fear & Greed data"""
        random.seed(datetime.now().day)
        value = random.randint(15, 85)
        
        if value <= 25:
            classification = 'Extreme Fear'
            emoji = '😱'
            color = '#FF0000'
        elif value <= 45:
            classification = 'Fear'
            emoji = '😟'
            color = '#FF6600'
        elif value <= 55:
            classification = 'Neutral'
            emoji = '😐'
            color = '#FFCC00'
        elif value <= 75:
            classification = 'Greed'
            emoji = '😃'
            color = '#66CC00'
        else:
            classification = 'Extreme Greed'
            emoji = '🤑'
            color = '#00CC00'
        
        return self._format_fear_greed_response(value, 'crypto', classification, emoji, color)
    
    def _generate_demo_stock_fear_greed(self) -> Dict[str, Any]:
        """Generate demo stock Fear & Greed data"""
        random.seed(datetime.now().day + 1)
        value = random.randint(20, 80)
        
        if value <= 25:
            classification = 'Extreme Fear'
            emoji = '😱'
            color = '#FF0000'
        elif value <= 45:
            classification = 'Fear'
            emoji = '😟'
            color = '#FF6600'
        elif value <= 55:
            classification = 'Neutral'
            emoji = '😐'
            color = '#FFCC00'
        elif value <= 75:
            classification = 'Greed'
            emoji = '😃'
            color = '#66CC00'
        else:
            classification = 'Extreme Greed'
            emoji = '🤑'
            color = '#00CC00'
        
        return self._format_fear_greed_response(value, 'stocks', classification, emoji, color)
    
    def _generate_demo_options_fear_greed(self) -> Dict[str, Any]:
        """Generate demo options sentiment data"""
        random.seed(datetime.now().day + 2)
        
        put_call_ratio = random.uniform(0.6, 1.4)
        
        if put_call_ratio > 1.2:
            classification = 'Extreme Fear'
            value = random.randint(10, 25)
            emoji = '😱'
            color = '#FF0000'
        elif put_call_ratio > 1.0:
            classification = 'Fear'
            value = random.randint(26, 45)
            emoji = '😟'
            color = '#FF6600'
        elif put_call_ratio > 0.9:
            classification = 'Neutral'
            value = random.randint(46, 55)
            emoji = '😐'
            color = '#FFCC00'
        elif put_call_ratio > 0.8:
            classification = 'Greed'
            value = random.randint(56, 75)
            emoji = '😃'
            color = '#66CC00'
        else:
            classification = 'Extreme Greed'
            value = random.randint(76, 90)
            emoji = '🤑'
            color = '#00CC00'
        
        response = self._format_fear_greed_response(value, 'options', classification, emoji, color)
        response['put_call_ratio'] = round(put_call_ratio, 2)
        response['implied_volatility_rank'] = random.randint(20, 80)
        
        return response
    
    def _format_fear_greed_response(self, value: int, asset_class: str, 
                                     classification: str, emoji: str = '😐', 
                                     color: str = '#FFCC00') -> Dict[str, Any]:
        """Format Fear & Greed response"""
        recommendation = self._get_trading_recommendation(value, asset_class)
        
        return {
            'asset_class': asset_class,
            'value': value,
            'classification': classification,
            'emoji': emoji,
            'color': color,
            'recommendation': recommendation,
            'components': self._get_index_components(asset_class),
            'demo_mode': self.demo_mode,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_trading_recommendation(self, value: int, asset_class: str) -> str:
        """Get trading recommendation based on index value"""
        if value <= 25:
            return f"BUYING OPPORTUNITY - {asset_class.capitalize()} showing extreme fear. Often a good entry point for contrarian investors."
        elif value <= 45:
            return f"CAUTIOUS BUY - {asset_class.capitalize()} in fear zone. Good for accumulation with dollar-cost averaging."
        elif value <= 55:
            return f"NEUTRAL - {asset_class.capitalize()} sentiment balanced. Wait for clearer signals."
        elif value <= 75:
            return f"TAKE PROFITS - {asset_class.capitalize()} showing greed. Consider securing gains on profitable positions."
        else:
            return f"HIGH RISK - {asset_class.capitalize()} in extreme greed. Be cautious with new positions."
    
    def _get_index_components(self, asset_class: str) -> Dict[str, float]:
        """Get components that make up the index"""
        random.seed(datetime.now().day + hash(asset_class))
        
        if asset_class == 'crypto':
            return {
                'volatility': round(random.uniform(0, 100), 1),
                'market_momentum': round(random.uniform(0, 100), 1),
                'social_media': round(random.uniform(0, 100), 1),
                'surveys': round(random.uniform(0, 100), 1),
                'dominance': round(random.uniform(0, 100), 1),
                'trends': round(random.uniform(0, 100), 1)
            }
        elif asset_class == 'stocks':
            return {
                'market_momentum': round(random.uniform(0, 100), 1),
                'stock_price_strength': round(random.uniform(0, 100), 1),
                'stock_price_breadth': round(random.uniform(0, 100), 1),
                'put_call_options': round(random.uniform(0, 100), 1),
                'junk_bond_demand': round(random.uniform(0, 100), 1),
                'safe_haven_demand': round(random.uniform(0, 100), 1),
                'market_volatility': round(random.uniform(0, 100), 1)
            }
        else:
            return {
                'put_call_ratio': round(random.uniform(0, 100), 1),
                'implied_volatility': round(random.uniform(0, 100), 1),
                'options_volume': round(random.uniform(0, 100), 1),
                'skew': round(random.uniform(0, 100), 1)
            }
    
    def _calculate_overall_sentiment(self) -> Dict[str, Any]:
        """Calculate overall market sentiment across all asset classes"""
        crypto_index = self.get_crypto_fear_greed()
        stock_index = self.get_stock_fear_greed()
        options_index = self.get_options_fear_greed()
        
        avg_value = (crypto_index['value'] + stock_index['value'] + options_index['value']) / 3
        
        if avg_value <= 25:
            overall_sentiment = 'Extreme Fear'
            emoji = '😱'
        elif avg_value <= 45:
            overall_sentiment = 'Fear'
            emoji = '😟'
        elif avg_value <= 55:
            overall_sentiment = 'Neutral'
            emoji = '😐'
        elif avg_value <= 75:
            overall_sentiment = 'Greed'
            emoji = '😃'
        else:
            overall_sentiment = 'Extreme Greed'
            emoji = '🤑'
        
        return {
            'value': round(avg_value, 1),
            'classification': overall_sentiment,
            'emoji': emoji,
            'message': f'Overall market sentiment across crypto, stocks, and options: {overall_sentiment}'
        }
