"""Gmail client using Replit's OAuth connector."""

import os
import json
import base64
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from email import message_from_bytes
from email.utils import parsedate_to_datetime


class GmailClient:
    """Client for reading Gmail emails using Replit's OAuth connector."""
    
    NEWSLETTER_SENDERS = [
        "tldr@tldrnewsletter.com",
        "noreply@barchart.com",
        "barchart",
        "investing.com",
        "noreply@webull.com",
        "webull",
        "newsletter@investing.com",
        "alerts@investing.com",
        "partner@barchart.com"
    ]
    
    def __init__(self):
        self.base_url = "https://gmail.googleapis.com/gmail/v1"
        self._access_token = None
        self._token_expires_at = None
    
    def _refresh_access_token(self) -> str:
        """Refresh OAuth access token from Replit connector."""
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        hostname = os.environ.get('REPLIT_CONNECTORS_HOSTNAME')
        if not hostname:
            raise ValueError('REPLIT_CONNECTORS_HOSTNAME not set')
        
        repl_identity = os.environ.get('REPL_IDENTITY')
        web_repl_renewal = os.environ.get('WEB_REPL_RENEWAL')
        
        if repl_identity:
            x_replit_token = f'repl {repl_identity}'
        elif web_repl_renewal:
            x_replit_token = f'depl {web_repl_renewal}'
        else:
            raise ValueError('Replit identity token not found')
        
        response = requests.get(
            f'https://{hostname}/api/v2/connection?include_secrets=true&connector_names=google-mail',
            headers={
                'Accept': 'application/json',
                'X_REPLIT_TOKEN': x_replit_token
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f'Failed to get Gmail connection: {response.status_code}')
        
        data = response.json()
        connection = data.get('items', [{}])[0]
        settings = connection.get('settings', {})
        
        self._access_token = settings.get('access_token') or settings.get('oauth', {}).get('credentials', {}).get('access_token')
        
        expires_at = settings.get('expires_at')
        if expires_at:
            self._token_expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00')).replace(tzinfo=None)
        else:
            self._token_expires_at = datetime.now() + timedelta(hours=1)
        
        if not self._access_token:
            raise ValueError('Gmail access token not found. Please reconnect Gmail.')
        
        return self._access_token
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """Make authenticated request to Gmail API."""
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
            print(f"Gmail API error: {response.status_code} - {response.text[:200]}")
            return {}
        
        return response.json()
    
    def test_connection(self) -> Dict:
        """Test the Gmail connection."""
        try:
            profile = self._make_request('/users/me/profile')
            return {
                'connected': True,
                'email': profile.get('emailAddress', 'Unknown'),
                'messages_total': profile.get('messagesTotal', 0)
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def list_labels(self) -> List[Dict]:
        """List all Gmail labels."""
        result = self._make_request('/users/me/labels')
        return result.get('labels', [])
    
    def search_messages(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for messages matching a query."""
        params = {
            'q': query,
            'maxResults': max_results
        }
        
        result = self._make_request('/users/me/messages', params=params)
        return result.get('messages', [])
    
    def get_message(self, message_id: str, format: str = 'full') -> Dict:
        """Get a specific message by ID."""
        params = {'format': format}
        return self._make_request(f'/users/me/messages/{message_id}', params=params)
    
    def get_message_body(self, message: Dict) -> str:
        """Extract the body text from a message."""
        payload = message.get('payload', {})
        
        if 'body' in payload and payload['body'].get('data'):
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        
        parts = payload.get('parts', [])
        body_text = ""
        
        for part in parts:
            mime_type = part.get('mimeType', '')
            
            if mime_type == 'text/plain':
                if part.get('body', {}).get('data'):
                    body_text += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            elif mime_type == 'text/html':
                if part.get('body', {}).get('data') and not body_text:
                    html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    body_text = self._html_to_text(html)
            elif 'parts' in part:
                for subpart in part['parts']:
                    if subpart.get('mimeType') == 'text/plain' and subpart.get('body', {}).get('data'):
                        body_text += base64.urlsafe_b64decode(subpart['body']['data']).decode('utf-8', errors='ignore')
        
        return body_text
    
    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion."""
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def get_message_headers(self, message: Dict) -> Dict[str, str]:
        """Extract headers from a message."""
        headers = {}
        payload = message.get('payload', {})
        
        for header in payload.get('headers', []):
            name = header.get('name', '').lower()
            value = header.get('value', '')
            headers[name] = value
        
        return headers
    
    def get_newsletter_emails(self, days: int = 30, max_results: int = 100) -> List[Dict]:
        """Get emails from known newsletter senders."""
        sender_queries = [f'from:{sender}' for sender in self.NEWSLETTER_SENDERS]
        query = f"({' OR '.join(sender_queries)}) newer_than:{days}d"
        
        print(f"Searching for newsletters: {query[:100]}...")
        
        message_refs = self.search_messages(query, max_results)
        print(f"Found {len(message_refs)} newsletter emails")
        
        emails = []
        for ref in message_refs:
            message = self.get_message(ref['id'])
            if message:
                headers = self.get_message_headers(message)
                body = self.get_message_body(message)
                
                emails.append({
                    'id': message.get('id'),
                    'thread_id': message.get('threadId'),
                    'from': headers.get('from', ''),
                    'subject': headers.get('subject', ''),
                    'date': headers.get('date', ''),
                    'body': body,
                    'snippet': message.get('snippet', '')
                })
        
        return emails
    
    def get_emails_by_sender(self, sender: str, days: int = 30, max_results: int = 50) -> List[Dict]:
        """Get emails from a specific sender."""
        query = f"from:{sender} newer_than:{days}d"
        
        message_refs = self.search_messages(query, max_results)
        
        emails = []
        for ref in message_refs:
            message = self.get_message(ref['id'])
            if message:
                headers = self.get_message_headers(message)
                body = self.get_message_body(message)
                
                emails.append({
                    'id': message.get('id'),
                    'from': headers.get('from', ''),
                    'subject': headers.get('subject', ''),
                    'date': headers.get('date', ''),
                    'body': body,
                    'snippet': message.get('snippet', '')
                })
        
        return emails
