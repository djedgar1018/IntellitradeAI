"""
Enhanced Crypto Data Fetcher
Hybrid approach: Yahoo Finance for historical data + CoinMarketCap for current prices
"""

import yfinance as yf
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time

class CryptoDataFetcher:
    """Fetch crypto data using hybrid approach"""
    
    def __init__(self):
        self.cmc_api_key = os.environ.get('COINMARKETCAP_API_KEY')
        self.cmc_base_url = "https://pro-api.coinmarketcap.com/v1"
        
        # Yahoo Finance crypto symbols
        self.yahoo_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'LTC': 'LTC-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'DOGE': 'DOGE-USD'
        }
    
    def fetch_historical_data(self, symbol, period='1y', interval='1d'):
        """
        Fetch historical OHLCV data using Yahoo Finance
        
        Args:
            symbol: Crypto symbol (BTC, ETH, LTC)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"📊 Fetching historical data for {symbol}...")
            
            # Get Yahoo Finance symbol
            yf_symbol = self.yahoo_symbols.get(symbol, f'{symbol}-USD')
            
            # Fetch data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"⚠️ No data found for {yf_symbol}")
                return None
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'adj close': 'adj_close'})
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Retrieved {len(df)} data points for {symbol}")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Latest close: ${df['close'].iloc[-1]:,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def fetch_current_price(self, symbol):
        """
        Fetch current price from CoinMarketCap API
        
        Args:
            symbol: Crypto symbol (BTC, ETH, LTC)
            
        Returns:
            Dict with current price data
        """
        try:
            if not self.cmc_api_key:
                print("⚠️ CoinMarketCap API key not found, using Yahoo Finance")
                return self._fetch_current_price_yahoo(symbol)
            
            print(f"💰 Fetching current price for {symbol}...")
            
            url = f"{self.cmc_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.cmc_api_key,
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
                    
                    price_data = {
                        'symbol': symbol,
                        'price': quote['price'],
                        'volume_24h': quote['volume_24h'],
                        'percent_change_1h': quote.get('percent_change_1h', 0),
                        'percent_change_24h': quote.get('percent_change_24h', 0),
                        'percent_change_7d': quote.get('percent_change_7d', 0),
                        'market_cap': quote.get('market_cap', 0),
                        'last_updated': quote.get('last_updated', datetime.now().isoformat())
                    }
                    
                    print(f"✅ Current {symbol} price: ${price_data['price']:,.2f}")
                    print(f"   24h change: {price_data['percent_change_24h']:+.2f}%")
                    
                    return price_data
                else:
                    print(f"⚠️ No data for {symbol}, falling back to Yahoo Finance")
                    return self._fetch_current_price_yahoo(symbol)
            else:
                print(f"⚠️ API error ({response.status_code}), using Yahoo Finance")
                return self._fetch_current_price_yahoo(symbol)
                
        except Exception as e:
            print(f"⚠️ Error with CoinMarketCap ({str(e)}), using Yahoo Finance")
            return self._fetch_current_price_yahoo(symbol)
    
    def _fetch_current_price_yahoo(self, symbol):
        """Fallback: Fetch current price from Yahoo Finance"""
        try:
            yf_symbol = self.yahoo_symbols.get(symbol, f'{symbol}-USD')
            ticker = yf.Ticker(yf_symbol)
            
            # Get latest data
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            previous_close = ticker.info.get('previousClose', latest['Close'])
            
            price_data = {
                'symbol': symbol,
                'price': float(latest['Close']),
                'volume_24h': float(hist['Volume'].sum()),
                'percent_change_24h': ((latest['Close'] - previous_close) / previous_close * 100),
                'last_updated': datetime.now().isoformat()
            }
            
            print(f"✅ Current {symbol} price (Yahoo): ${price_data['price']:,.2f}")
            return price_data
            
        except Exception as e:
            print(f"❌ Error fetching Yahoo Finance data: {str(e)}")
            return None
    
    def fetch_multiple_symbols(self, symbols, period='1y'):
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of crypto symbols
            period: Time period
            
        Returns:
            Dict of {symbol: DataFrame}
        """
        print(f"\n📈 Fetching data for {len(symbols)} symbols...")
        print("=" * 60)
        
        results = {}
        
        for symbol in symbols:
            df = self.fetch_historical_data(symbol, period=period)
            
            if df is not None:
                results[symbol] = df
            
            # Rate limiting (be respectful to APIs)
            time.sleep(0.5)
        
        print("=" * 60)
        print(f"✅ Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        
        return results
    
    def save_to_cache(self, data_dict, filename='crypto_cache.json'):
        """Save fetched data to cache"""
        try:
            cache_data = {}
            
            for symbol, df in data_dict.items():
                cache_data[symbol] = {
                    'data': df.to_dict(orient='index'),
                    'last_updated': datetime.now().isoformat(),
                    'rows': len(df)
                }
            
            cache_path = f'data/{filename}'
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            print(f"💾 Data cached to: {cache_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving cache: {str(e)}")
            return False
    
    def get_data_summary(self, df):
        """Get summary statistics for dataset"""
        if df is None or df.empty:
            return None
        
        summary = {
            'total_days': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            },
            'price': {
                'latest': float(df['close'].iloc[-1]),
                'highest': float(df['close'].max()),
                'lowest': float(df['close'].min()),
                'average': float(df['close'].mean())
            },
            'returns': {
                'total': float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100),
                'daily_avg': float(df['close'].pct_change().mean() * 100)
            },
            'volatility': {
                'daily_std': float(df['close'].pct_change().std() * 100)
            }
        }
        
        return summary


# Example usage
if __name__ == '__main__':
    print("\n🚀 Crypto Data Fetcher - Hybrid Mode")
    print("   • Historical: Yahoo Finance (FREE)")
    print("   • Current: CoinMarketCap API (when available)")
    print("=" * 60)
    
    fetcher = CryptoDataFetcher()
    
    # Test with BTC
    symbols = ['BTC', 'ETH', 'LTC']
    
    # Fetch historical data
    data = fetcher.fetch_multiple_symbols(symbols, period='6mo')
    
    # Show summaries
    print("\n📊 Data Summaries:")
    print("=" * 60)
    for symbol, df in data.items():
        summary = fetcher.get_data_summary(df)
        if summary:
            print(f"\n{symbol}:")
            print(f"  Total Days: {summary['total_days']}")
            print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            print(f"  Current Price: ${summary['price']['latest']:,.2f}")
            print(f"  Total Return: {summary['returns']['total']:+.2f}%")
            print(f"  Daily Volatility: {summary['volatility']['daily_std']:.2f}%")
    
    # Save cache
    if data:
        fetcher.save_to_cache(data)
    
    # Test current price
    print("\n💰 Current Prices:")
    print("=" * 60)
    for symbol in symbols[:2]:  # Test just 2 to save API credits
        current = fetcher.fetch_current_price(symbol)
        time.sleep(1)  # Rate limiting
    
    print("\n✅ Data fetching complete!")
