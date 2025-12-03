"""
Email Subscription Manager for IntelliTradeAI
Manages email subscriptions to market data sources
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailSubscriptionManager:
    """
    Manages email subscriptions to market data providers
    Note: Actual subscriptions require manual signup on respective websites
    """
    
    def __init__(self):
        self.subscription_sources = {
            'investing.com': {
                'url': 'https://www.investing.com/',
                'name': 'Investing.com',
                'description': 'Real-time financial news and market data',
                'categories': ['stocks', 'crypto', 'forex', 'commodities']
            },
            'barchart.com': {
                'url': 'https://www.barchart.com/',
                'name': 'Barchart',
                'description': 'Futures, stocks, and options market data',
                'categories': ['futures', 'stocks', 'options', 'etfs']
            },
            'webull.com': {
                'url': 'https://www.webull.com/',
                'name': 'Webull',
                'description': 'Stock trading platform with market insights',
                'categories': ['stocks', 'options', 'crypto', 'news']
            },
            'coindesk.com': {
                'url': 'https://www.coindesk.com/',
                'name': 'CoinDesk',
                'description': 'Cryptocurrency news and analysis',
                'categories': ['crypto', 'blockchain', 'defi']
            },
            'marketwatch.com': {
                'url': 'https://www.marketwatch.com/',
                'name': 'MarketWatch',
                'description': 'Stock market news and financial insights',
                'categories': ['stocks', 'markets', 'economy']
            },
            'bloomberg.com': {
                'url': 'https://www.bloomberg.com/',
                'name': 'Bloomberg',
                'description': 'Global financial news and data',
                'categories': ['stocks', 'crypto', 'markets', 'economy']
            }
        }
        
        self.active_subscriptions = []
        self.subscription_history = []
    
    def get_available_sources(self) -> Dict[str, Any]:
        """Get list of available subscription sources"""
        return {
            'sources': self.subscription_sources,
            'total_sources': len(self.subscription_sources),
            'recommended_sources': ['investing.com', 'barchart.com', 'webull.com', 'coindesk.com']
        }
    
    def get_subscription_instructions(self, source: str) -> Dict[str, Any]:
        """
        Get instructions for subscribing to a source
        
        Args:
            source: Source identifier (e.g., 'investing.com')
            
        Returns:
            Subscription instructions
        """
        if source not in self.subscription_sources:
            return {
                'success': False,
                'error': f'Source {source} not recognized',
                'available_sources': list(self.subscription_sources.keys())
            }
        
        source_info = self.subscription_sources[source]
        
        return {
            'source': source,
            'name': source_info['name'],
            'url': source_info['url'],
            'description': source_info['description'],
            'categories': source_info['categories'],
            'subscription_steps': self._get_subscription_steps(source),
            'estimated_time': '2-3 minutes',
            'note': 'These are manual subscriptions requiring email verification on the respective websites'
        }
    
    def _get_subscription_steps(self, source: str) -> List[str]:
        """Get step-by-step subscription instructions"""
        if source == 'investing.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Click "Sign Up" or "Register" in the top right corner',
                '3. Enter your email address and create a password',
                '4. Verify your email address through the confirmation link',
                '5. Navigate to "Settings" > "Email Notifications"',
                '6. Enable notifications for: Market Updates, Price Alerts, Daily Newsletter',
                '7. Save your notification preferences'
            ]
        
        elif source == 'barchart.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Click "Sign Up Free" button',
                '3. Complete the registration form with your email',
                '4. Confirm your email address',
                '5. Go to "My Barchart" > "Email Preferences"',
                '6. Subscribe to: Daily Market Summary, Options Activity, Futures Insights',
                '7. Confirm subscription preferences'
            ]
        
        elif source == 'webull.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Click "Sign Up" or download the Webull app',
                '3. Register with your email address',
                '4. Complete email verification',
                '5. In app/website, go to "Settings" > "Notifications"',
                '6. Enable: Market News, Price Alerts, Daily Digest',
                '7. Save notification settings'
            ]
        
        elif source == 'coindesk.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Scroll to footer and find "Newsletter" section',
                '3. Enter your email address',
                '4. Select newsletter types: The Node, Market Wrap, Weekend Reads',
                '5. Click "Subscribe"',
                '6. Confirm subscription through email verification'
            ]
        
        elif source == 'marketwatch.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Click "Sign In/Register" in the top navigation',
                '3. Create account with your email',
                '4. Go to "Settings" > "Email Alerts"',
                '5. Enable: Market Summary, Breaking News, Top Stories',
                '6. Save alert preferences'
            ]
        
        elif source == 'bloomberg.com':
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Click "Subscribe" or "Sign Up"',
                '3. Create Bloomberg account',
                '4. Navigate to "Account Settings" > "Email Preferences"',
                '5. Select newsletters: Markets, Technology, Crypto',
                '6. Confirm preferences'
            ]
        
        else:
            return [
                f'1. Visit {self.subscription_sources[source]["url"]}',
                '2. Look for "Sign Up" or "Newsletter" link',
                '3. Enter your email address',
                '4. Confirm subscription through verification email',
                '5. Configure notification preferences'
            ]
    
    def mark_subscription_completed(self, source: str, email: str) -> Dict[str, Any]:
        """
        Mark a subscription as completed
        
        Args:
            source: Source identifier
            email: Email used for subscription
            
        Returns:
            Confirmation status
        """
        if source not in self.subscription_sources:
            return {
                'success': False,
                'error': f'Source {source} not recognized'
            }
        
        subscription = {
            'source': source,
            'email': email,
            'subscribed_at': datetime.now().isoformat(),
            'status': 'active',
            'categories': self.subscription_sources[source]['categories']
        }
        
        existing = next(
            (s for s in self.active_subscriptions if s['source'] == source and s['email'] == email),
            None
        )
        
        if existing:
            return {
                'success': True,
                'message': f'Subscription to {source} already exists',
                'subscription': existing
            }
        
        self.active_subscriptions.append(subscription)
        self.subscription_history.append({
            **subscription,
            'action': 'subscribed'
        })
        
        return {
            'success': True,
            'message': f'Successfully marked subscription to {source}',
            'subscription': subscription,
            'total_active_subscriptions': len(self.active_subscriptions)
        }
    
    def get_active_subscriptions(self) -> Dict[str, Any]:
        """Get all active subscriptions"""
        return {
            'active_subscriptions': self.active_subscriptions,
            'total_subscriptions': len(self.active_subscriptions),
            'sources': [s['source'] for s in self.active_subscriptions],
            'categories': list(set(
                cat for sub in self.active_subscriptions 
                for cat in sub['categories']
            ))
        }
    
    def get_subscription_recommendations(self, asset_focus: str = 'all') -> Dict[str, Any]:
        """
        Get recommended subscriptions based on asset focus
        
        Args:
            asset_focus: 'stocks', 'crypto', 'options', or 'all'
            
        Returns:
            Recommended sources for the asset class
        """
        recommendations = []
        
        for source, info in self.subscription_sources.items():
            if asset_focus == 'all' or asset_focus in info['categories']:
                recommendations.append({
                    'source': source,
                    'name': info['name'],
                    'description': info['description'],
                    'relevance': 'high' if asset_focus in info['categories'][:2] else 'medium',
                    'categories': info['categories']
                })
        
        recommendations.sort(key=lambda x: x['relevance'], reverse=True)
        
        return {
            'asset_focus': asset_focus,
            'recommendations': recommendations,
            'total_recommended': len(recommendations),
            'priority_sources': [r['source'] for r in recommendations if r['relevance'] == 'high']
        }
    
    def unsubscribe(self, source: str, email: str) -> Dict[str, Any]:
        """Mark a subscription as cancelled"""
        subscription = next(
            (s for s in self.active_subscriptions if s['source'] == source and s['email'] == email),
            None
        )
        
        if not subscription:
            return {
                'success': False,
                'error': f'No active subscription found for {source} with email {email}'
            }
        
        self.active_subscriptions.remove(subscription)
        
        self.subscription_history.append({
            **subscription,
            'action': 'unsubscribed',
            'unsubscribed_at': datetime.now().isoformat()
        })
        
        source_info = self.subscription_sources[source]
        
        return {
            'success': True,
            'message': f'Subscription to {source} marked as cancelled',
            'unsubscribe_instructions': [
                f'1. Check your email inbox for newsletters from {source_info["name"]}',
                '2. Open any recent newsletter email',
                '3. Scroll to the bottom of the email',
                '4. Click the "Unsubscribe" link',
                '5. Confirm unsubscription on the website'
            ],
            'note': 'You need to manually unsubscribe through the email link'
        }
    
    def get_subscription_summary(self) -> Dict[str, Any]:
        """Get complete subscription summary"""
        category_coverage = {}
        
        for sub in self.active_subscriptions:
            for cat in sub['categories']:
                category_coverage[cat] = category_coverage.get(cat, 0) + 1
        
        return {
            'total_active': len(self.active_subscriptions),
            'total_sources': len(self.subscription_sources),
            'subscription_coverage': f"{len(self.active_subscriptions)}/{len(self.subscription_sources)}",
            'category_coverage': category_coverage,
            'active_sources': [s['source'] for s in self.active_subscriptions],
            'unsubscribed_count': len([h for h in self.subscription_history if h.get('action') == 'unsubscribed']),
            'recommendation': self._get_coverage_recommendation(len(self.active_subscriptions))
        }
    
    def _get_coverage_recommendation(self, active_count: int) -> str:
        """Get recommendation based on subscription coverage"""
        total = len(self.subscription_sources)
        coverage_percent = (active_count / total) * 100
        
        if coverage_percent >= 75:
            return "EXCELLENT - You have comprehensive market data coverage across multiple sources"
        elif coverage_percent >= 50:
            return "GOOD - Solid coverage, consider adding 1-2 more sources for complete insights"
        elif coverage_percent >= 25:
            return "FAIR - Basic coverage, recommend subscribing to additional sources for better data"
        else:
            return "LOW - Limited coverage, strongly recommend subscribing to key sources (investing.com, barchart.com, webull.com)"
