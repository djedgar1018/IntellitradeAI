"""
Data Ingestion Module
Handles API calls for cryptocurrency and stock data
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
        
    def fetch_crypto_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch cryptocurrency data from CoinMarketCap API
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
            period: Time period for data
            interval: Data interval
            
        Returns:
            crypto_data: Dictionary with crypto data for each symbol
        """
        try:
            crypto_data = {}
            
            for symbol in symbols:
                try:
                    # Get symbol ID from CoinMarketCap
                    symbol_id = self._get_crypto_symbol_id(symbol)
                    
                    if symbol_id:
                        # Fetch historical data
                        historical_data = self._fetch_crypto_historical_data(symbol_id, period)
                        
                        if historical_data is not None:
                            crypto_data[symbol] = historical_data
                        else:
                            # Fallback to current price data
                            current_data = self._fetch_crypto_current_price(symbol)
                            if current_data:
                                crypto_data[symbol] = current_data
                    
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            raise Exception(f"Error fetching crypto data: {str(e)}")
    
    def _get_crypto_symbol_id(self, symbol):
        """Get CoinMarketCap symbol ID"""
        try:
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/map"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'symbol': symbol,
                'limit': 1
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    return data['data'][0]['id']
            
            return None
            
        except Exception as e:
            print(f"Error getting symbol ID for {symbol}: {str(e)}")
            return None
    
    def _fetch_crypto_historical_data(self, symbol_id, period):
        """Fetch historical crypto data from CoinMarketCap"""
        try:
            # Calculate date range
            end_date = datetime.now()
            if period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=365)
            
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/ohlcv/historical"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'id': symbol_id,
                'time_start': start_date.strftime('%Y-%m-%d'),
                'time_end': end_date.strftime('%Y-%m-%d'),
                'interval': 'daily'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    quotes = data['data']['quotes']
                    
                    # Convert to DataFrame
                    records = []
                    for quote in quotes:
                        record = {
                            'date': pd.to_datetime(quote['time_open']),
                            'open': quote['quote']['USD']['open'],
                            'high': quote['quote']['USD']['high'],
                            'low': quote['quote']['USD']['low'],
                            'close': quote['quote']['USD']['close'],
                            'volume': quote['quote']['USD']['volume']
                        }
                        records.append(record)
                    
                    df = pd.DataFrame(records)
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    return df
            
            return None
            
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return None
    
    def _fetch_crypto_current_price(self, symbol):
        """Fetch current crypto price from CoinMarketCap"""
        try:
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and symbol in data['data']:
                    quote = data['data'][symbol]['quote']['USD']
                    
                    # Create a simple DataFrame with current price
                    current_data = {
                        'open': [quote['price']],
                        'high': [quote['price']],
                        'low': [quote['price']],
                        'close': [quote['price']],
                        'volume': [quote['volume_24h']]
                    }
                    
                    df = pd.DataFrame(current_data, index=[datetime.now()])
                    return df
            
            return None
            
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {str(e)}")
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
            
            # Fetch crypto data
            if crypto_symbols:
                crypto_data = self.fetch_crypto_data(crypto_symbols, period, interval)
                mixed_data.update(crypto_data)
            
            # Fetch stock data
            if stock_symbols:
                stock_data = self.fetch_stock_data(stock_symbols, period, interval)
                mixed_data.update(stock_data)
            
            return mixed_data
            
        except Exception as e:
            raise Exception(f"Error fetching mixed data: {str(e)}")
    
    def get_real_time_quotes(self, symbols, data_type='mixed'):
        """
        Get real-time quotes for specified symbols
        
        Args:
            symbols: List of symbols
            data_type: Type of data ('crypto', 'stock', 'mixed')
            
        Returns:
            quotes: Dictionary with real-time quotes
        """
        try:
            quotes = {}
            
            if data_type in ['crypto', 'mixed']:
                # Get crypto quotes
                crypto_symbols = [s for s in symbols if s in config.DATA_CONFIG["crypto_symbols"]]
                if crypto_symbols:
                    crypto_quotes = self._get_crypto_quotes(crypto_symbols)
                    quotes.update(crypto_quotes)
            
            if data_type in ['stock', 'mixed']:
                # Get stock quotes
                stock_symbols = [s for s in symbols if s in config.DATA_CONFIG["stock_symbols"]]
                if stock_symbols:
                    stock_quotes = self._get_stock_quotes(stock_symbols)
                    quotes.update(stock_quotes)
            
            return quotes
            
        except Exception as e:
            raise Exception(f"Error getting real-time quotes: {str(e)}")
    
    def _get_crypto_quotes(self, symbols):
        """Get real-time crypto quotes"""
        try:
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'symbol': ','.join(symbols),
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                quotes = {}
                
                if 'data' in data:
                    for symbol in symbols:
                        if symbol in data['data']:
                            quote_data = data['data'][symbol]['quote']['USD']
                            quotes[symbol] = {
                                'price': quote_data['price'],
                                'change_24h': quote_data['percent_change_24h'],
                                'volume_24h': quote_data['volume_24h'],
                                'market_cap': quote_data['market_cap'],
                                'last_updated': data['data'][symbol]['last_updated']
                            }
                
                return quotes
            
            return {}
            
        except Exception as e:
            print(f"Error getting crypto quotes: {str(e)}")
            return {}
    
    def _get_stock_quotes(self, symbols):
        """Get real-time stock quotes"""
        try:
            quotes = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if 'regularMarketPrice' in info:
                        quotes[symbol] = {
                            'price': info.get('regularMarketPrice', 0),
                            'change_24h': info.get('regularMarketChangePercent', 0),
                            'volume_24h': info.get('regularMarketVolume', 0),
                            'market_cap': info.get('marketCap', 0),
                            'last_updated': datetime.now().isoformat()
                        }
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error getting quote for {symbol}: {str(e)}")
                    continue
            
            return quotes
            
        except Exception as e:
            print(f"Error getting stock quotes: {str(e)}")
            return {}