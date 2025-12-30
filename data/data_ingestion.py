"""
Data Ingestion Module - Hybrid Approach
- Yahoo Finance: Historical OHLCV data (free, reliable)
- CoinMarketCap: Current prices and real-time data (paid API, more accurate)
"""

import yfinance as yf
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
from config import config

class DataIngestion:
    """Class for ingesting financial data from various APIs"""
    
    def __init__(self):
        self.coinmarketcap_api_key = config.COINMARKETCAP_API_KEY
        self.coinmarketcap_base_url = config.COINMARKETCAP_BASE_URL
        self.yahoo_timeout = config.YAHOO_FINANCE_TIMEOUT
    
    def _get_yahoo_crypto_map(self):
        """Comprehensive Yahoo Finance symbol mapping for 100+ cryptocurrencies"""
        return {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'USDT': 'USDT-USD',
            'XRP': 'XRP-USD',
            'BNB': 'BNB-USD',
            'SOL': 'SOL-USD',
            'USDC': 'USDC-USD',
            'TRX': 'TRX-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'AVAX': 'AVAX-USD',
            'DOT': 'DOT-USD',
            'NEAR': 'NEAR-USD',
            'ICP': 'ICP-USD',
            'APT': 'APT21794-USD',
            'SUI': 'SUI20947-USD',
            'SEI': 'SEI-USD',
            'ATOM': 'ATOM-USD',
            'ALGO': 'ALGO-USD',
            'FTM': 'FTM-USD',
            'HBAR': 'HBAR-USD',
            'EOS': 'EOS-USD',
            'XLM': 'XLM-USD',
            'MATIC': 'MATIC-USD',
            'ARB': 'ARB11841-USD',
            'OP': 'OP-USD',
            'IMX': 'IMX10603-USD',
            'MNT': 'MNT27075-USD',
            'STRK': 'STRK-USD',
            'ZK': 'ZK-USD',
            'METIS': 'METIS-USD',
            'DAI': 'DAI-USD',
            'FDUSD': 'FDUSD-USD',
            'TUSD': 'TUSD-USD',
            'USDD': 'USDD-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI7083-USD',
            'AAVE': 'AAVE-USD',
            'MKR': 'MKR-USD',
            'LDO': 'LDO-USD',
            'INJ': 'INJ-USD',
            'CRV': 'CRV-USD',
            'SNX': 'SNX-USD',
            'COMP': 'COMP-USD',
            'SUSHI': 'SUSHI-USD',
            '1INCH': '1INCH-USD',
            'DYDX': 'DYDX-USD',
            'JUP': 'JUP29210-USD',
            'PENDLE': 'PENDLE-USD',
            'FET': 'FET-USD',
            'RNDR': 'RENDER-USD',
            'AGIX': 'AGIX-USD',
            'OCEAN': 'OCEAN-USD',
            'TAO': 'TAO22974-USD',
            'AKT': 'AKT-USD',
            'ARKM': 'ARKM-USD',
            'WLD': 'WLD-USD',
            'AI': 'AI-USD',
            'VIRTUAL': 'VIRTUAL-USD',
            'AI16Z': 'AI16Z-USD',
            'GOAT': 'GOAT-USD',
            'AIXBT': 'AIXBT-USD',
            'ZEREBRO': 'ZEREBRO-USD',
            'GRIFFAIN': 'GRIFFAIN-USD',
            'FARTCOIN': 'FARTCOIN-USD',
            'GRASS': 'GRASS-USD',
            'IO': 'IO-USD',
            'NOS': 'NOS-USD',
            'PRIME': 'PRIME-USD',
            'LUNA2': 'LUNA2-USD',
            'ARC': 'ARC-USD',
            'SWARMS': 'SWARMS-USD',
            'PAAL': 'PAAL-USD',
            'CGPT': 'CGPT-USD',
            'AGRS': 'AGRS-USD',
            'SHIB': 'SHIB-USD',
            'PEPE': 'PEPE24478-USD',
            'WIF': 'WIF-USD',
            'BONK': 'BONK-USD',
            'FLOKI': 'FLOKI-USD',
            'MEME': 'MEME28301-USD',
            'ELON': 'ELON-USD',
            'BRETT': 'BRETT29743-USD',
            'POPCAT': 'POPCAT-USD',
            'MOG': 'MOG-USD',
            'TURBO': 'TURBO-USD',
            'NEIRO': 'NEIRO-USD',
            'SPX': 'SPX-USD',
            'PNUT': 'PNUT-USD',
            'MYRO': 'MYRO-USD',
            'MEW': 'MEW-USD',
            'BABYDOGE': 'BABYDOGE-USD',
            'COQ': 'COQ-USD',
            'BOME': 'BOME-USD',
            'SLERF': 'SLERF-USD',
            'PONKE': 'PONKE-USD',
            'GIGA': 'GIGA-USD',
            'MOODENG': 'MOODENG-USD',
            'APE': 'APE18876-USD',
            'BLUR': 'BLUR-USD',
            'ENS': 'ENS-USD',
            'LOOKS': 'LOOKS-USD',
            'X2Y2': 'X2Y2-USD',
            'RARE': 'RARE-USD',
            'JPEG': 'JPEG-USD',
            'BEND': 'BEND-USD',
            'SUDO': 'SUDO-USD',
            'NFT': 'NFT-USD',
            'MAGIC': 'MAGIC-USD',
            'PENGU': 'PENGU-USD',
            'ONDO': 'ONDO-USD',
            'CFG': 'CFG-USD',
            'MPL': 'MPL-USD',
            'CPOOL': 'CPOOL-USD',
            'SAND': 'SAND-USD',
            'AXS': 'AXS-USD',
            'MANA': 'MANA-USD',
            'GALA': 'GALA-USD',
            'ENJ': 'ENJ-USD',
            'ILV': 'ILV-USD',
            'BEAM': 'BEAM28298-USD',
            'RONIN': 'RONIN-USD',
            'GRT': 'GRT6719-USD',
            'FIL': 'FIL-USD',
            'AR': 'AR-USD',
            'STX': 'STX4847-USD',
            'PYTH': 'PYTH-USD',
            'API3': 'API3-USD',
            'OKB': 'OKB-USD',
            'CRO': 'CRO-USD',
            'LEO': 'LEO-USD',
            'KCS': 'KCS-USD',
            'GT': 'GT-USD',
            'HT': 'HT-USD',
            'XMR': 'XMR-USD',
            'ZEC': 'ZEC-USD',
            'DASH': 'DASH-USD',
            'STETH': 'STETH-USD',
            'RETH': 'RETH-USD',
            'CBETH': 'CBETH-USD',
            'RPL': 'RPL-USD',
            'LTC': 'LTC-USD',
            'BCH': 'BCH-USD',
            'ETC': 'ETC-USD',
            'VET': 'VET-USD',
            'TON': 'TON11419-USD'
        }
        
    def fetch_crypto_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch cryptocurrency data using Yahoo Finance for historical OHLCV
        (Works with both free and paid CoinMarketCap plans)
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH', 'XRP'])
            period: Time period for data
            interval: Data interval
            
        Returns:
            crypto_data: Dictionary with crypto data for each symbol
        """
        try:
            crypto_data = {}
            
            yahoo_crypto_map = self._get_yahoo_crypto_map()
            
            for symbol in symbols:
                try:
                    # Get Yahoo Finance symbol
                    yf_symbol = yahoo_crypto_map.get(symbol, f'{symbol}-USD')
                    
                    # Fetch historical data using Yahoo Finance
                    ticker = yf.Ticker(yf_symbol)
                    hist_data = ticker.history(period=period, interval=interval)
                    
                    if not hist_data.empty:
                        # Standardize column names
                        hist_data.columns = [col.lower() for col in hist_data.columns]
                        
                        # Select only OHLCV data
                        ohlcv_data = hist_data[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        # Enrich with current CoinMarketCap data (if available)
                        try:
                            current_price = self._fetch_crypto_current_price(symbol)
                            if current_price:
                                # Update the latest price with CMC data (more accurate)
                                ohlcv_data.iloc[-1, ohlcv_data.columns.get_loc('close')] = current_price['price']
                                print(f"✅ Fetched {len(ohlcv_data)} data points for {symbol} (enriched with CMC)")
                            else:
                                print(f"✅ Fetched {len(ohlcv_data)} data points for {symbol}")
                        except:
                            print(f"✅ Fetched {len(ohlcv_data)} data points for {symbol}")
                        
                        crypto_data[symbol] = ohlcv_data
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                    else:
                        print(f"⚠️ No data found for {symbol} ({yf_symbol})")
                    
                except Exception as e:
                    print(f"❌ Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            raise Exception(f"Error fetching crypto data: {str(e)}")
    
    def _fetch_crypto_current_price(self, symbol):
        """
        Fetch current crypto price from CoinMarketCap (using paid API)
        Falls back gracefully if API unavailable
        """
        try:
            if not self.coinmarketcap_api_key:
                return None
            
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key,
                'Accept': 'application/json'
            }
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and symbol in data['data']:
                    quote = data['data'][symbol]['quote']['USD']
                    return {
                        'symbol': symbol,
                        'price': quote['price'],
                        'volume_24h': quote.get('volume_24h', 0),
                        'percent_change_24h': quote.get('percent_change_24h', 0),
                        'market_cap': quote.get('market_cap', 0)
                    }
            
            return None
            
        except Exception as e:
            # Fail silently - Yahoo data is already good enough
            return None
    
    def fetch_stock_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            period: Time period for data
            interval: Data interval
            
        Returns:
            stock_data: Dictionary with stock data for each symbol
        """
        try:
            stock_data = {}
            
            # Validate symbols input
            if not symbols:
                return stock_data
            
            for symbol in symbols:
                try:
                    # Create ticker object
                    ticker = yf.Ticker(symbol)
                    
                    # Fetch historical data
                    hist_data = ticker.history(period=period, interval=interval)
                    
                    if not hist_data.empty:
                        # Rename columns to match our standard format
                        hist_data.columns = [col.lower() for col in hist_data.columns]
                        
                        # Select only OHLCV data
                        ohlcv_data = hist_data[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        stock_data[symbol] = ohlcv_data
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error fetching stock data for {symbol}: {str(e)}")
                    continue
            
            return stock_data
            
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def fetch_mixed_data(self, crypto_symbols=None, stock_symbols=None, period='1y', interval='1d'):
        """
        Fetch both crypto and stock data
        
        Args:
            crypto_symbols: List of crypto symbols
            stock_symbols: List of stock symbols
            period: Time period for data
            interval: Data interval
            
        Returns:
            mixed_data: Dictionary with both crypto and stock data
        """
        try:
            mixed_data = {}
            
            # Use default symbols if not provided
            if crypto_symbols is None:
                crypto_symbols = config.DATA_CONFIG["crypto_symbols"]
            if stock_symbols is None:
                stock_symbols = config.DATA_CONFIG["stock_symbols"]
            
            # Fetch crypto data (Yahoo historical + CMC current)
            if crypto_symbols:
                crypto_data = self.fetch_crypto_data(crypto_symbols, period, interval)
                mixed_data.update(crypto_data)
            
            # Fetch stock data (Yahoo Finance)
            if stock_symbols:
                stock_data = self.fetch_stock_data(stock_symbols, period, interval)
                mixed_data.update(stock_data)
            
            return mixed_data
            
        except Exception as e:
            raise Exception(f"Error fetching mixed data: {str(e)}")
    
    def get_real_time_quotes(self, symbols, data_type='mixed'):
        """
        Get real-time quotes using CoinMarketCap API (paid plan)
        
        Args:
            symbols: List of symbols
            data_type: Type of data ('crypto', 'stock', 'mixed')
            
        Returns:
            quotes: Dictionary with real-time quotes
        """
        try:
            quotes = {}
            
            for symbol in symbols:
                try:
                    # For crypto, use CoinMarketCap
                    if data_type in ['crypto', 'mixed']:
                        price_data = self._fetch_crypto_current_price(symbol)
                        if price_data:
                            quotes[symbol] = price_data
                    
                    # For stocks, use Yahoo Finance
                    if data_type in ['stock', 'mixed']:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        if 'regularMarketPrice' in info:
                            quotes[symbol] = {
                                'symbol': symbol,
                                'price': info['regularMarketPrice'],
                                'volume': info.get('volume', 0),
                                'market_cap': info.get('marketCap', 0)
                            }
                    
                except Exception as e:
                    print(f"Error fetching quote for {symbol}: {str(e)}")
                    continue
            
            return quotes
            
        except Exception as e:
            raise Exception(f"Error fetching real-time quotes: {str(e)}")
    
    def get_crypto_metadata(self, symbols):
        """
        Get cryptocurrency metadata from CoinMarketCap
        
        Args:
            symbols: List of crypto symbols
            
        Returns:
            metadata: Dictionary with metadata for each symbol
        """
        try:
            if not self.coinmarketcap_api_key:
                return {}
            
            metadata = {}
            
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/info"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key,
                'Accept': 'application/json'
            }
            params = {
                'symbol': ','.join(symbols)
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    for symbol in symbols:
                        if symbol in data['data']:
                            info = data['data'][symbol]
                            metadata[symbol] = {
                                'name': info.get('name', ''),
                                'symbol': info.get('symbol', ''),
                                'category': info.get('category', ''),
                                'description': info.get('description', ''),
                                'logo': info.get('logo', ''),
                                'website': info.get('urls', {}).get('website', []),
                                'twitter': info.get('urls', {}).get('twitter', [])
                            }
            
            return metadata
            
        except Exception as e:
            print(f"Error fetching metadata: {str(e)}")
            return {}
