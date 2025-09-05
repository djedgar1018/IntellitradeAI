"""
Secure Blockchain Wallet Management
Handles wallet creation, encryption, and blockchain interactions
"""
import secrets
import hashlib
import json
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from web3 import Web3
from eth_account import Account
import qrcode
from io import BytesIO

class SecureWalletManager:
    """
    Enterprise-grade wallet manager with military-level encryption
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.w3 = Web3()  # For Ethereum operations
        
    def _generate_master_key(self) -> str:
        """Generate cryptographically secure master key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
    
    def _derive_encryption_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count for security
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def create_ethereum_wallet(self, user_password: str) -> Dict[str, str]:
        """
        Create secure Ethereum wallet with encrypted private key storage
        """
        # Generate new account
        account = Account.create()
        
        # Generate salt for encryption
        salt = os.urandom(16)
        
        # Encrypt private key
        encryption_key = self._derive_encryption_key(user_password, salt)
        fernet = Fernet(encryption_key)
        
        encrypted_private_key = fernet.encrypt(account.key.hex().encode())
        encrypted_mnemonic = fernet.encrypt(Account.create_with_mnemonic()[1].encode())
        
        return {
            'address': account.address,
            'private_key_encrypted': encrypted_private_key.decode(),
            'mnemonic_encrypted': encrypted_mnemonic.decode(),
            'salt': base64.b64encode(salt).decode(),
            'public_key': account.address,
            'derivation_path': "m/44'/60'/0'/0/0"
        }
    
    def decrypt_private_key(self, encrypted_key: str, password: str, salt: str) -> str:
        """Decrypt private key for transaction signing"""
        try:
            salt_bytes = base64.b64decode(salt.encode())
            encryption_key = self._derive_encryption_key(password, salt_bytes)
            fernet = Fernet(encryption_key)
            
            decrypted_key = fernet.decrypt(encrypted_key.encode())
            return decrypted_key.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt private key: {str(e)}")
    
    def get_wallet_balance(self, address: str, token_address: Optional[str] = None) -> float:
        """
        Get wallet balance (ETH or ERC-20 token)
        """
        try:
            # Connect to Ethereum mainnet (you can switch to testnet for development)
            infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY', '')}"
            w3 = Web3(Web3.HTTPProvider(infura_url))
            
            if not w3.is_connected():
                raise ConnectionError("Cannot connect to Ethereum network")
            
            if token_address:
                # Get ERC-20 token balance
                # This would require implementing ERC-20 contract interaction
                # For now, return 0 as placeholder
                return 0.0
            else:
                # Get ETH balance
                balance_wei = w3.eth.get_balance(address)
                balance_eth = w3.from_wei(balance_wei, 'ether')
                return float(balance_eth)
                
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0
    
    def generate_wallet_qr_code(self, address: str) -> bytes:
        """Generate QR code for wallet address"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(address)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        byte_stream = BytesIO()
        img.save(byte_stream, format='PNG')
        return byte_stream.getvalue()
    
    def create_transaction(self, from_address: str, to_address: str, 
                          amount: float, private_key: str, 
                          gas_price: Optional[int] = None) -> Dict[str, str]:
        """
        Create and sign Ethereum transaction
        """
        try:
            infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY', '')}"
            w3 = Web3(Web3.HTTPProvider(infura_url))
            
            # Get nonce
            nonce = w3.eth.get_transaction_count(from_address)
            
            # Get current gas price if not specified
            if gas_price is None:
                gas_price = w3.eth.gas_price
            
            # Build transaction
            transaction = {
                'to': to_address,
                'value': w3.to_wei(amount, 'ether'),
                'gas': 21000,  # Standard gas limit for ETH transfer
                'gasPrice': gas_price,
                'nonce': nonce,
            }
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            
            return {
                'signed_transaction': signed_txn.rawTransaction.hex(),
                'transaction_hash': signed_txn.hash.hex(),
                'gas_estimate': str(21000 * gas_price),
                'status': 'ready_to_broadcast'
            }
            
        except Exception as e:
            return {
                'error': f"Transaction creation failed: {str(e)}",
                'status': 'failed'
            }
    
    def broadcast_transaction(self, signed_transaction: str) -> Dict[str, str]:
        """Broadcast signed transaction to blockchain"""
        try:
            infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY', '')}"
            w3 = Web3(Web3.HTTPProvider(infura_url))
            
            tx_hash = w3.eth.send_raw_transaction(signed_transaction)
            
            return {
                'transaction_hash': tx_hash.hex(),
                'status': 'broadcasted',
                'explorer_url': f"https://etherscan.io/tx/{tx_hash.hex()}"
            }
            
        except Exception as e:
            return {
                'error': f"Broadcast failed: {str(e)}",
                'status': 'failed'
            }

class PortfolioTracker:
    """
    Track and calculate portfolio performance
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def calculate_portfolio_performance(self, portfolio_id: int) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        
        # Get portfolio data
        portfolio_query = """
        SELECT starting_capital, current_value, total_invested, realized_pnl 
        FROM portfolios WHERE id = %s
        """
        
        holdings_query = """
        SELECT symbol, quantity, average_price, current_price, market_value 
        FROM holdings WHERE portfolio_id = %s
        """
        
        trades_query = """
        SELECT pnl, fees, executed_at 
        FROM trades WHERE portfolio_id = %s AND status = 'CLOSED'
        ORDER BY executed_at
        """
        
        # Execute queries (placeholder - implement with your DB connection)
        portfolio_data = self._execute_query(portfolio_query, (portfolio_id,))
        holdings = self._execute_query(holdings_query, (portfolio_id,))
        trades = self._execute_query(trades_query, (portfolio_id,))
        
        if not portfolio_data:
            return {}
        
        portfolio = portfolio_data[0]
        
        # Calculate metrics
        total_return = portfolio['current_value'] - portfolio['starting_capital']
        total_return_pct = (total_return / portfolio['starting_capital']) * 100 if portfolio['starting_capital'] > 0 else 0
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(holding['market_value'] - (holding['quantity'] * holding['average_price']) 
                            for holding in holdings)
        
        # Calculate win rate from closed trades
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [trade['pnl'] for trade in trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        return_std = self._calculate_std(returns) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        return {
            'total_value': portfolio['current_value'],
            'total_invested': portfolio['total_invested'],
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'realized_pnl': portfolio['realized_pnl'],
            'unrealized_pnl': unrealized_pnl,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'number_of_holdings': len(holdings)
        }
    
    def _calculate_std(self, values):
        """Calculate standard deviation"""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _execute_query(self, query, params):
        """Placeholder for database query execution"""
        # This would be implemented with your actual database connection
        # For now, return empty results
        return []

class SecurityManager:
    """
    Handle all security-related operations
    """
    
    @staticmethod
    def generate_api_credentials() -> Tuple[str, str]:
        """Generate secure API key and secret"""
        api_key = 'ak_' + secrets.token_urlsafe(32)
        api_secret = 'as_' + secrets.token_urlsafe(64)
        return api_key, api_secret
    
    @staticmethod
    def hash_password(password: str) -> Tuple[str, str]:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode('utf-8'),
                                          salt.encode('utf-8'),
                                          100000)
        return base64.b64encode(password_hash).decode('utf-8') == hashed
    
    @staticmethod
    def generate_2fa_secret() -> str:
        """Generate 2FA secret key"""
        return secrets.token_hex(16)
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(64)