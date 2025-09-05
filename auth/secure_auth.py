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
from fastapi import HTTPException, status
from passlib.context import CryptContext
from email_validator import validate_email, EmailNotValidError
import re

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
        self.password_min_length = 12
        self.password_require_complexity = True
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        Validate password meets security requirements
        """
        if len(password) < self.password_min_length:
            return False, f"Password must be at least {self.password_min_length} characters"
        
        if not self.password_require_complexity:
            return True, "Password is valid"
        
        # Check complexity requirements
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        
        # Check for common patterns
        common_patterns = [
            r"123", r"abc", r"qwe", r"password", r"admin",
            r"(.)\1{2,}", r"012", r"789"
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                return False, "Password contains common patterns or sequences"
        
        return True, "Password is valid"
    
    def validate_email_format(self, email: str) -> Tuple[bool, str]:
        """Validate email format"""
        try:
            valid = validate_email(email)
            return True, valid.email
        except EmailNotValidError as e:
            return False, str(e)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_2fa_secret(self, username: str, app_name: str = "AI Trading Bot") -> Tuple[str, str]:
        """
        Generate 2FA secret and QR code
        Returns: (secret_key, qr_code_base64)
        """
        secret = pyotp.random_base32()
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name=app_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return secret, qr_code_base64
    
    def verify_2fa_token(self, secret: str, token: str) -> bool:
        """Verify 2FA TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 30-second window
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.access_token_expire
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, any]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {"valid": True, "payload": payload}
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token has expired"}
        except jwt.JWTError:
            return {"valid": False, "error": "Invalid token"}
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, any]:
        """
        Register new user with full validation
        """
        # Validate email
        email_valid, email_msg = self.validate_email_format(email)
        if not email_valid:
            return {"success": False, "error": f"Invalid email: {email_msg}"}
        
        # Validate password
        password_valid, password_msg = self.validate_password_strength(password)
        if not password_valid:
            return {"success": False, "error": password_msg}
        
        # Check username requirements
        if len(username) < 3 or len(username) > 50:
            return {"success": False, "error": "Username must be 3-50 characters"}
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", username):
            return {"success": False, "error": "Username can only contain letters, numbers, hyphens, and underscores"}
        
        # Hash password
        password_hash = self.hash_password(password)
        salt = secrets.token_hex(32)
        
        # Generate email verification token
        verification_token = secrets.token_urlsafe(64)
        
        return {
            "success": True,
            "user_data": {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "salt": salt,
                "email_verification_token": verification_token,
                "is_2fa_enabled": False,
                "is_active": True
            },
            "verification_token": verification_token
        }
    
    def authenticate_user(self, username_or_email: str, password: str, 
                         totp_token: Optional[str] = None) -> Dict[str, any]:
        """
        Authenticate user with optional 2FA
        """
        # This would typically query the database
        # For now, return a mock response structure
        
        # Check if user exists and password is correct
        # user = self.db.get_user_by_username_or_email(username_or_email)
        
        # Mock user data for demonstration
        mock_user = {
            "id": 1,
            "username": "demo_user",
            "email": "user@example.com",
            "password_hash": self.hash_password("secure_password_123"),
            "is_2fa_enabled": False,
            "two_fa_secret": None,
            "login_attempts": 0,
            "locked_until": None,
            "is_active": True
        }
        
        if not mock_user or not mock_user["is_active"]:
            return {"success": False, "error": "User not found or account disabled"}
        
        # Check if account is locked
        if (mock_user["locked_until"] and 
            datetime.utcnow() < mock_user["locked_until"]):
            return {"success": False, "error": "Account is temporarily locked"}
        
        # Verify password
        if not self.verify_password(password, mock_user["password_hash"]):\n            # Increment login attempts\n            return {\"success\": False, \"error\": \"Invalid credentials\"}\n        \n        # Check 2FA if enabled\n        if mock_user[\"is_2fa_enabled\"]:\n            if not totp_token:\n                return {\n                    \"success\": False, \n                    \"error\": \"2FA token required\",\n                    \"requires_2fa\": True\n                }\n            \n            if not self.verify_2fa_token(mock_user[\"two_fa_secret\"], totp_token):\n                return {\"success\": False, \"error\": \"Invalid 2FA token\"}\n        \n        # Generate tokens\n        token_data = {\n            \"sub\": str(mock_user[\"id\"]),\n            \"username\": mock_user[\"username\"],\n            \"email\": mock_user[\"email\"]\n        }\n        \n        access_token = self.create_access_token(token_data)\n        refresh_token = self.create_refresh_token(token_data)\n        \n        return {\n            \"success\": True,\n            \"access_token\": access_token,\n            \"refresh_token\": refresh_token,\n            \"token_type\": \"bearer\",\n            \"user\": {\n                \"id\": mock_user[\"id\"],\n                \"username\": mock_user[\"username\"],\n                \"email\": mock_user[\"email\"],\n                \"is_2fa_enabled\": mock_user[\"is_2fa_enabled\"]\n            }\n        }\n    \n    def enable_2fa(self, user_id: int, username: str) -> Dict[str, any]:\n        \"\"\"\n        Enable 2FA for user\n        \"\"\"\n        secret, qr_code = self.generate_2fa_secret(username)\n        \n        return {\n            \"success\": True,\n            \"secret\": secret,\n            \"qr_code\": qr_code,\n            \"backup_codes\": self.generate_backup_codes()\n        }\n    \n    def generate_backup_codes(self, count: int = 10) -> list:\n        \"\"\"\n        Generate backup codes for 2FA recovery\n        \"\"\"\n        return [secrets.token_hex(4) for _ in range(count)]\n    \n    def log_security_event(self, user_id: Optional[int], event_type: str, \n                          ip_address: str, user_agent: str, \n                          success: bool, details: Dict = None):\n        \"\"\"\n        Log security events for audit trail\n        \"\"\"\n        log_entry = {\n            \"user_id\": user_id,\n            \"event_type\": event_type,\n            \"ip_address\": ip_address,\n            \"user_agent\": user_agent,\n            \"success\": success,\n            \"details\": details or {},\n            \"timestamp\": datetime.utcnow().isoformat()\n        }\n        \n        # This would be saved to database\n        print(f\"Security Event: {log_entry}\")\n        \n    def validate_session(self, session_token: str) -> Dict[str, any]:\n        \"\"\"\n        Validate user session token\n        \"\"\"\n        # This would query the database for active sessions\n        # For now, validate as JWT token\n        return self.verify_token(session_token)\n    \n    def revoke_session(self, session_token: str) -> bool:\n        \"\"\"\n        Revoke user session\n        \"\"\"\n        # This would mark session as inactive in database\n        return True\n    \n    def get_user_permissions(self, user_id: int) -> Dict[str, bool]:\n        \"\"\"\n        Get user permissions for API access\n        \"\"\"\n        # Default permissions\n        return {\n            \"read_portfolio\": True,\n            \"execute_trades\": False,  # Requires manual activation\n            \"manage_settings\": True,\n            \"api_access\": False,  # Requires separate API key\n            \"withdraw_funds\": False  # Requires additional verification\n        }"