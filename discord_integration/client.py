"""Discord client using Replit's OAuth connector for reading trade messages."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import requests


class DiscordClient:
    """Client for reading Discord messages using Replit's OAuth connector."""
    
    def __init__(self):
        self.base_url = "https://discord.com/api/v10"
        self._connection_settings = None
        self._access_token = None
        self._token_expires_at = None
        
    def _get_replit_token(self) -> str:
        """Get Replit identity token for connector API."""
        repl_identity = os.environ.get('REPL_IDENTITY')
        web_repl_renewal = os.environ.get('WEB_REPL_RENEWAL')
        
        if repl_identity:
            return f'repl {repl_identity}'
        elif web_repl_renewal:
            return f'depl {web_repl_renewal}'
        else:
            raise ValueError('Replit identity token not found')
    
    def _refresh_access_token(self) -> str:
        """Refresh OAuth access token from Replit connector."""
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        hostname = os.environ.get('REPLIT_CONNECTORS_HOSTNAME')
        if not hostname:
            raise ValueError('REPLIT_CONNECTORS_HOSTNAME not set')
        
        x_replit_token = self._get_replit_token()
        
        response = requests.get(
            f'https://{hostname}/api/v2/connection?include_secrets=true&connector_names=discord',
            headers={
                'Accept': 'application/json',
                'X_REPLIT_TOKEN': x_replit_token
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f'Failed to get Discord connection: {response.status_code}')
        
        data = response.json()
        connection = data.get('items', [{}])[0]
        settings = connection.get('settings', {})
        
        self._access_token = settings.get('access_token') or settings.get('oauth', {}).get('credentials', {}).get('access_token')
        
        expires_at = settings.get('expires_at')
        if expires_at:
            self._token_expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        else:
            self._token_expires_at = datetime.now() + timedelta(hours=1)
        
        if not self._access_token:
            raise ValueError('Discord access token not found. Please reconnect Discord.')
        
        return self._access_token
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """Make authenticated request to Discord API."""
        token = self._refresh_access_token()
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        else:
            response = requests.request(method, url, headers=headers, json=params)
        
        if response.status_code == 401:
            self._access_token = None
            self._token_expires_at = None
            return self._make_request(endpoint, method, params)
        
        if response.status_code != 200:
            print(f"Discord API error: {response.status_code} - {response.text}")
            return {}
        
        return response.json()
    
    def get_guilds(self) -> List[Dict]:
        """Get list of Discord servers (guilds) the bot has access to."""
        return self._make_request('/users/@me/guilds')
    
    def get_channels(self, guild_id: str) -> List[Dict]:
        """Get all channels in a guild."""
        return self._make_request(f'/guilds/{guild_id}/channels')
    
    def get_channel_messages(self, channel_id: str, limit: int = 100, before: str = None, after: str = None) -> List[Dict]:
        """Get messages from a channel or thread."""
        params = {'limit': min(limit, 100)}
        if before:
            params['before'] = before
        if after:
            params['after'] = after
        
        return self._make_request(f'/channels/{channel_id}/messages', params=params)
    
    def get_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict]:
        """Get all messages from a thread."""
        return self.get_channel_messages(thread_id, limit)
    
    def get_active_threads(self, guild_id: str) -> List[Dict]:
        """Get all active threads in a guild."""
        data = self._make_request(f'/guilds/{guild_id}/threads/active')
        return data.get('threads', [])
    
    def get_archived_threads(self, channel_id: str) -> List[Dict]:
        """Get archived threads in a channel."""
        data = self._make_request(f'/channels/{channel_id}/threads/archived/public')
        return data.get('threads', [])
    
    def get_message_history(self, channel_id: str, days: int = 365) -> List[Dict]:
        """Get message history for the past N days from a channel/thread."""
        all_messages = []
        before_id = None
        cutoff_date = datetime.now() - timedelta(days=days)
        
        while True:
            messages = self.get_channel_messages(channel_id, limit=100, before=before_id)
            
            if not messages:
                break
            
            for msg in messages:
                msg_time = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                if msg_time.replace(tzinfo=None) < cutoff_date:
                    return all_messages
                all_messages.append(msg)
            
            before_id = messages[-1]['id']
            
            if len(messages) < 100:
                break
        
        return all_messages
    
    def find_trading_channels(self, guild_id: str) -> List[Dict]:
        """Find channels that likely contain trading discussions."""
        trading_keywords = ['trade', 'trading', 'signal', 'alert', 'crypto', 'stock', 
                          'call', 'put', 'option', 'buy', 'sell', 'position']
        
        channels = self.get_channels(guild_id)
        trading_channels = []
        
        for channel in channels:
            channel_name = channel.get('name', '').lower()
            if any(keyword in channel_name for keyword in trading_keywords):
                trading_channels.append(channel)
        
        return trading_channels
    
    def test_connection(self) -> Dict:
        """Test the Discord connection and return user info."""
        try:
            user_data = self._make_request('/users/@me')
            guilds = self.get_guilds()
            return {
                'connected': True,
                'user': user_data.get('username', 'Unknown'),
                'guild_count': len(guilds) if isinstance(guilds, list) else 0,
                'guilds': [g.get('name') for g in guilds] if isinstance(guilds, list) else []
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
