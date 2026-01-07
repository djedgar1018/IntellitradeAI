"""IMAP client for reading Gmail emails with full inbox access."""

import os
import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import re


class IMAPClient:
    """Client for reading emails via IMAP with App Password authentication."""
    
    IMAP_SERVER = "imap.gmail.com"
    IMAP_PORT = 993
    
    NEWSLETTER_SENDERS = [
        "tldr",
        "barchart",
        "investing.com",
        "webull",
        "newsletter",
        "alerts"
    ]
    
    def __init__(self):
        self.email_address = os.environ.get('GMAIL_ADDRESS')
        self.app_password = os.environ.get('GMAIL_APP_PASSWORD')
        self._connection = None
    
    def connect(self) -> bool:
        """Connect to Gmail IMAP server."""
        if not self.email_address or not self.app_password:
            print("Gmail credentials not found in environment variables")
            return False
        
        try:
            self._connection = imaplib.IMAP4_SSL(self.IMAP_SERVER, self.IMAP_PORT)
            self._connection.login(self.email_address, self.app_password)
            return True
        except imaplib.IMAP4.error as e:
            print(f"IMAP login failed: {e}")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IMAP server."""
        if self._connection:
            try:
                self._connection.logout()
            except:
                pass
            self._connection = None
    
    def test_connection(self) -> Dict:
        """Test the IMAP connection."""
        if not self.email_address or not self.app_password:
            return {
                'connected': False,
                'error': 'Gmail credentials not configured. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD.'
            }
        
        try:
            if self.connect():
                self._connection.select('INBOX')
                status, messages = self._connection.search(None, 'ALL')
                total = len(messages[0].split()) if messages[0] else 0
                self.disconnect()
                
                return {
                    'connected': True,
                    'email': self.email_address,
                    'inbox_count': total
                }
            else:
                return {
                    'connected': False,
                    'error': 'Failed to connect to Gmail IMAP'
                }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def _decode_header_value(self, value: str) -> str:
        """Decode email header value."""
        if not value:
            return ""
        
        decoded_parts = decode_header(value)
        result = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                result += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                result += part
        return result
    
    def _get_email_body(self, msg) -> str:
        """Extract email body text."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body += payload.decode(charset, errors='ignore')
                    except:
                        pass
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html = payload.decode(charset, errors='ignore')
                            body = self._html_to_text(html)
                    except:
                        pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='ignore')
            except:
                pass
        
        return body
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
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
    
    def search_emails(self, query: str, folder: str = 'INBOX', max_results: int = 100) -> List[Dict]:
        """Search for emails matching criteria."""
        if not self.connect():
            return []
        
        try:
            self._connection.select(folder)
            
            status, messages = self._connection.search(None, query)
            
            if status != 'OK':
                return []
            
            email_ids = messages[0].split()
            email_ids = email_ids[-max_results:] if len(email_ids) > max_results else email_ids
            email_ids.reverse()
            
            emails = []
            for email_id in email_ids:
                status, msg_data = self._connection.fetch(email_id, '(RFC822)')
                
                if status != 'OK':
                    continue
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        subject = self._decode_header_value(msg.get('Subject', ''))
                        from_addr = self._decode_header_value(msg.get('From', ''))
                        date = msg.get('Date', '')
                        body = self._get_email_body(msg)
                        
                        emails.append({
                            'id': email_id.decode(),
                            'from': from_addr,
                            'subject': subject,
                            'date': date,
                            'body': body,
                            'snippet': body[:300] if body else ''
                        })
            
            return emails
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_newsletter_emails(self, days: int = 30, max_results: int = 100) -> List[Dict]:
        """Get emails from known newsletter senders."""
        if not self.connect():
            return []
        
        try:
            self._connection.select('INBOX')
            
            since_date = (datetime.now() - timedelta(days=days)).strftime('%d-%b-%Y')
            
            all_emails = []
            
            for sender in self.NEWSLETTER_SENDERS:
                query = f'(FROM "{sender}" SINCE {since_date})'
                
                try:
                    status, messages = self._connection.search(None, query)
                    
                    if status != 'OK' or not messages[0]:
                        continue
                    
                    email_ids = messages[0].split()
                    print(f"Found {len(email_ids)} emails from '{sender}'")
                    
                    for email_id in email_ids[-20:]:
                        try:
                            status, msg_data = self._connection.fetch(email_id, '(RFC822)')
                            
                            if status != 'OK':
                                continue
                            
                            for response_part in msg_data:
                                if isinstance(response_part, tuple):
                                    msg = email.message_from_bytes(response_part[1])
                                    
                                    subject = self._decode_header_value(msg.get('Subject', ''))
                                    from_addr = self._decode_header_value(msg.get('From', ''))
                                    date = msg.get('Date', '')
                                    body = self._get_email_body(msg)
                                    
                                    all_emails.append({
                                        'id': email_id.decode(),
                                        'from': from_addr,
                                        'subject': subject,
                                        'date': date,
                                        'body': body,
                                        'snippet': body[:300] if body else ''
                                    })
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    print(f"Error searching for {sender}: {e}")
                    continue
            
            all_emails = all_emails[:max_results]
            
            return all_emails
            
        except Exception as e:
            print(f"Newsletter fetch error: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_emails_by_sender(self, sender: str, days: int = 30, max_results: int = 50) -> List[Dict]:
        """Get emails from a specific sender."""
        since_date = (datetime.now() - timedelta(days=days)).strftime('%d-%b-%Y')
        query = f'(FROM "{sender}" SINCE {since_date})'
        return self.search_emails(query, max_results=max_results)
