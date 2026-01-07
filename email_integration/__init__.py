"""Email integration for extracting trading signals from newsletters."""

from email_integration.gmail_client import GmailClient
from email_integration.imap_client import IMAPClient
from email_integration.newsletter_parser import NewsletterParser
from email_integration.email_service import EmailTradingService

__all__ = [
    'GmailClient',
    'IMAPClient',
    'NewsletterParser', 
    'EmailTradingService'
]
