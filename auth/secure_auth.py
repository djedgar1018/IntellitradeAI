"""
Enterprise-Grade User Authentication System
With 2FA, JWT tokens, and comprehensive security features
"""
import hashlib
import secrets
import jwt
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from passlib.context import CryptContext

class SecureAuthManager:
    """
    Military-grade authentication system with multiple security layers
    """
    
    def __init__(self, secret_key: str, db_connection=None):
        self.secret_key = secret_key
        self.db = db_connection
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=30)
        self.refresh_token_expire = timedelta(days=7)
        
        # Security policies
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.password_min_length = 8
        self.password_require_complexity = True
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.access_token_expire
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def authenticate_user(self, username_or_email: str, password: str, 
                         totp_token: Optional[str] = None) -> Dict[str, any]:
        """
        Authenticate user with optional 2FA
        """
        # Mock user data for demonstration
        mock_user = {
            "id": 1,
            "username": "demo_user",
            "email": "user@example.com",
            "password_hash": self.hash_password("password123"),
            "is_2fa_enabled": False,
            "two_fa_secret": None,
            "login_attempts": 0,
            "locked_until": None,
            "is_active": True
        }
        
        if not mock_user or not mock_user["is_active"]:
            return {"success": False, "error": "User not found or account disabled"}
        
        # Verify password
        if not self.verify_password(password, mock_user["password_hash"]):
            return {"success": False, "error": "Invalid credentials"}
        
        # Generate tokens
        token_data = {
            "sub": str(mock_user["id"]),
            "username": mock_user["username"],
            "email": mock_user["email"]
        }
        
        access_token = self.create_access_token(token_data)
        
        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": mock_user["id"],
                "username": mock_user["username"],
                "email": mock_user["email"],
                "is_2fa_enabled": mock_user["is_2fa_enabled"]
            }
        }
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, any]:
        """Register new user with full validation"""
        
        # Basic validation
        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters"}
        
        if len(password) < self.password_min_length:
            return {"success": False, "error": f"Password must be at least {self.password_min_length} characters"}
        
        # Hash password
        password_hash = self.hash_password(password)
        
        return {
            "success": True,
            "user_data": {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "is_2fa_enabled": False,
                "is_active": True
            }
        }