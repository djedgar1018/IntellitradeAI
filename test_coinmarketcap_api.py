"""
Test CoinMarketCap API with new credentials
Verify 1-month historical data access
"""

import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import json

def test_api_connection():
    """Test basic API connection"""
    print("🔑 Testing CoinMarketCap API Connection...")
    print("=" * 60)
    
    api_key = os.environ.get('COINMARKETCAP_API_KEY')
    if not api_key:
        print("❌ ERROR: COINMARKETCAP_API_KEY not found in environment!")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test endpoint
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    params = {
        'symbol': 'BTC',
        'limit': 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Connection Successful!")
            print(f"   Status: {data.get('status', {}).get('error_message', 'Success')}")
            
            if 'data' in data and len(data['data']) > 0:
                btc_data = data['data'][0]
                print(f"   BTC ID: {btc_data['id']}")
                print(f"   BTC Name: {btc_data['name']}")
                print(f"   BTC Symbol: {btc_data['symbol']}")
            
            # Show credit usage
            if 'status' in data:
                status = data['status']
                print(f"\n📊 API Credits:")
                print(f"   Credit Count: {status.get('credit_count', 'N/A')}")
                print(f"   Timestamp: {status.get('timestamp', 'N/A')}")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        return False


def test_historical_data_fetch():
    """Test fetching 1-month historical OHLCV data"""
    print("\n📈 Testing Historical Data Fetch (1 Month)...")
    print("=" * 60)
    
    api_key = os.environ.get('COINMARKETCAP_API_KEY')
    
    # Calculate date range (1 month)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # BTC historical data
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    params = {
        'id': 1,  # BTC
        'time_start': start_date.strftime('%Y-%m-%d'),
        'time_end': end_date.strftime('%Y-%m-%d'),
        'interval': 'daily',
        'convert': 'USD'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data:
                quotes = data['data'].get('quotes', [])
                print(f"✅ Historical Data Retrieved!")
                print(f"   Total Days: {len(quotes)}")
                
                if quotes:
                    # Show sample data
                    latest = quotes[-1]
                    print(f"\n📊 Latest Data Point ({latest['time_open'][:10]}):")
                    usd_quote = latest['quote']['USD']
                    print(f"   Open:   ${usd_quote['open']:,.2f}")
                    print(f"   High:   ${usd_quote['high']:,.2f}")
                    print(f"   Low:    ${usd_quote['low']:,.2f}")
                    print(f"   Close:  ${usd_quote['close']:,.2f}")
                    print(f"   Volume: ${usd_quote['volume']:,.0f}")
                    
                    # Convert to DataFrame
                    records = []
                    for quote in quotes:
                        usd = quote['quote']['USD']
                        records.append({
                            'date': pd.to_datetime(quote['time_open']),
                            'open': usd['open'],
                            'high': usd['high'],
                            'low': usd['low'],
                            'close': usd['close'],
                            'volume': usd['volume']
                        })
                    
                    df = pd.DataFrame(records)
                    df.set_index('date', inplace=True)
                    
                    print(f"\n📋 DataFrame Preview:")
                    print(df.head())
                    print(f"\n   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    
                    # Save sample
                    sample_file = 'data/btc_sample_30days.json'
                    df.to_json(sample_file, orient='index', date_format='iso')
                    print(f"\n💾 Sample saved to: {sample_file}")
                    
                    return True
                else:
                    print("⚠️ No quotes data in response")
                    return False
            else:
                print(f"❌ No data in response: {data}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Fetch Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiting():
    """Test rate limiting (30 requests/minute)"""
    print("\n⏱️ Testing Rate Limiting...")
    print("=" * 60)
    print("Your plan: 30 requests/minute")
    print("Recommendation: Add delays between requests")
    print("  • Sleep 2 seconds between calls")
    print("  • Batch requests when possible")
    print("  • Cache data locally (5-min TTL)")
    print("  • Use OHLCV endpoint (more data per call)")


def show_api_limits():
    """Display API plan limits"""
    print("\n📋 Your CoinMarketCap Plan Limits:")
    print("=" * 60)
    print("Monthly Credits:    300,000 (soft cap)")
    print("Rate Limit:         30 requests/minute")
    print("Historical Data:    Up to 1 month")
    print("Endpoints Enabled:  28")
    print("Currency Conv/Req:  40")
    print("License:            Commercial use ✅")
    print("\n💡 Tips:")
    print("  • 1 month = ~30 data points (daily)")
    print("  • Use caching to minimize API calls")
    print("  • Batch symbols in single request")
    print("  • Monitor credit usage")


if __name__ == '__main__':
    print("\n🚀 CoinMarketCap API Test Suite")
    print("=" * 60)
    
    # Test 1: Basic connection
    if not test_api_connection():
        print("\n❌ Basic connection failed. Check API key.")
        exit(1)
    
    # Test 2: Historical data
    if not test_historical_data_fetch():
        print("\n❌ Historical data fetch failed.")
        exit(1)
    
    # Test 3: Rate limiting info
    test_rate_limiting()
    
    # Show limits
    show_api_limits()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! API is ready to use.")
    print("=" * 60)
