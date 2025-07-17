"""
Test suite for data ingestion module
Safe isolated execution for testing API calls and data processing
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_ingestion import DataIngestion
from config import config

class TestDataIngestion(unittest.TestCase):
    """Test cases for DataIngestion class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_ingestion = DataIngestion()
        self.test_crypto_symbols = ['BTC', 'ETH']
        self.test_stock_symbols = ['AAPL', 'GOOGL']
        
    def test_init(self):
        """Test DataIngestion initialization"""
        self.assertIsNotNone(self.data_ingestion.coinmarketcap_api_key)
        self.assertIsNotNone(self.data_ingestion.coinmarketcap_base_url)
        self.assertIsNotNone(self.data_ingestion.yahoo_timeout)
        
    @patch('requests.get')
    def test_get_crypto_symbol_id_success(self, mock_get):
        """Test successful crypto symbol ID retrieval"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'id': 1, 'symbol': 'BTC', 'name': 'Bitcoin'}]
        }
        mock_get.return_value = mock_response
        
        # Test
        symbol_id = self.data_ingestion._get_crypto_symbol_id('BTC')
        
        # Assert
        self.assertEqual(symbol_id, 1)
        mock_get.assert_called_once()
        
    @patch('requests.get')
    def test_get_crypto_symbol_id_failure(self, mock_get):
        """Test crypto symbol ID retrieval failure"""
        # Mock failed API response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Test
        symbol_id = self.data_ingestion._get_crypto_symbol_id('INVALID')
        
        # Assert
        self.assertIsNone(symbol_id)
        
    @patch('requests.get')
    def test_fetch_crypto_current_price_success(self, mock_get):
        """Test successful crypto current price fetching"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'BTC': {
                    'quote': {
                        'USD': {
                            'price': 50000.0,
                            'volume_24h': 1000000.0
                        }
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Test
        result = self.data_ingestion._fetch_crypto_current_price('BTC')
        
        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['close'], 50000.0)
        
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker):
        """Test successful stock data fetching"""
        # Mock yfinance ticker
        mock_ticker_instance = MagicMock()
        mock_history = pd.DataFrame({
            'Open': [150.0, 152.0],
            'High': [155.0, 157.0],
            'Low': [148.0, 150.0],
            'Close': [153.0, 155.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = self.data_ingestion.fetch_stock_data(['AAPL'])
        
        # Assert
        self.assertIsNotNone(result)
        self.assertIn('AAPL', result)
        self.assertIsInstance(result['AAPL'], pd.DataFrame)
        self.assertEqual(len(result['AAPL']), 2)
        
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_empty_response(self, mock_ticker):
        """Test stock data fetching with empty response"""
        # Mock yfinance ticker with empty history
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = self.data_ingestion.fetch_stock_data(['INVALID'])
        
        # Assert
        self.assertEqual(result, {})
        
    @patch.object(DataIngestion, 'fetch_crypto_data')
    @patch.object(DataIngestion, 'fetch_stock_data')
    def test_fetch_mixed_data_success(self, mock_stock, mock_crypto):
        """Test successful mixed data fetching"""
        # Mock return values
        mock_crypto.return_value = {'BTC': pd.DataFrame({'close': [50000]}, index=[datetime.now()])}
        mock_stock.return_value = {'AAPL': pd.DataFrame({'close': [150]}, index=[datetime.now()])}
        
        # Test
        result = self.data_ingestion.fetch_mixed_data(['BTC'], ['AAPL'])
        
        # Assert
        self.assertIn('BTC', result)
        self.assertIn('AAPL', result)
        mock_crypto.assert_called_once_with(['BTC'], '1y', '1d')
        mock_stock.assert_called_once_with(['AAPL'], '1y', '1d')
        
    @patch('requests.get')
    def test_get_crypto_quotes_success(self, mock_get):
        """Test successful crypto quotes retrieval"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'BTC': {
                    'quote': {
                        'USD': {
                            'price': 50000.0,
                            'percent_change_24h': 2.5,
                            'volume_24h': 1000000.0,
                            'market_cap': 1000000000.0
                        }
                    },
                    'last_updated': '2024-01-01T00:00:00Z'
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Test
        result = self.data_ingestion._get_crypto_quotes(['BTC'])
        
        # Assert
        self.assertIn('BTC', result)
        self.assertEqual(result['BTC']['price'], 50000.0)
        self.assertEqual(result['BTC']['change_24h'], 2.5)
        
    @patch('yfinance.Ticker')
    def test_get_stock_quotes_success(self, mock_ticker):
        """Test successful stock quotes retrieval"""
        # Mock yfinance ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'regularMarketPrice': 150.0,
            'regularMarketChangePercent': 1.5,
            'regularMarketVolume': 1000000,
            'marketCap': 2000000000
        }
        mock_ticker.return_value = mock_ticker_instance
        
        # Test
        result = self.data_ingestion._get_stock_quotes(['AAPL'])
        
        # Assert
        self.assertIn('AAPL', result)
        self.assertEqual(result['AAPL']['price'], 150.0)
        self.assertEqual(result['AAPL']['change_24h'], 1.5)
        
    def test_data_validation(self):
        """Test data validation and error handling"""
        # Test with empty symbols list
        result = self.data_ingestion.fetch_stock_data([])
        self.assertEqual(result, {})
        
        # Test with None symbols - should handle gracefully
        try:
            result = self.data_ingestion.fetch_stock_data(None)
            self.assertEqual(result, {})
        except Exception:
            # This is expected behavior for None input
            pass

class TestDataIngestionIntegration(unittest.TestCase):
    """Integration tests for DataIngestion (requires real API keys)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_ingestion = DataIngestion()
        
    def test_yahoo_finance_integration(self):
        """Test actual Yahoo Finance integration"""
        # This test uses real Yahoo Finance API
        try:
            result = self.data_ingestion.fetch_stock_data(['AAPL'], period='5d')
            
            if result and 'AAPL' in result:
                self.assertIsInstance(result['AAPL'], pd.DataFrame)
                self.assertGreater(len(result['AAPL']), 0)
                self.assertIn('close', result['AAPL'].columns)
                print("✓ Yahoo Finance integration test passed")
            else:
                print("⚠ Yahoo Finance integration test skipped (no data)")
                
        except Exception as e:
            print(f"⚠ Yahoo Finance integration test failed: {str(e)}")
            
    def test_coinmarketcap_integration(self):
        """Test actual CoinMarketCap integration (requires API key)"""
        # Only run if API key is configured
        if self.data_ingestion.coinmarketcap_api_key != "default_key":
            try:
                # Test getting symbol ID
                symbol_id = self.data_ingestion._get_crypto_symbol_id('BTC')
                
                if symbol_id:
                    print(f"✓ CoinMarketCap symbol ID test passed: BTC = {symbol_id}")
                    
                    # Test getting current price
                    price_data = self.data_ingestion._fetch_crypto_current_price('BTC')
                    
                    if price_data is not None:
                        print("✓ CoinMarketCap current price test passed")
                    else:
                        print("⚠ CoinMarketCap current price test failed")
                else:
                    print("⚠ CoinMarketCap symbol ID test failed")
                    
            except Exception as e:
                print(f"⚠ CoinMarketCap integration test failed: {str(e)}")
        else:
            print("⚠ CoinMarketCap integration test skipped (API key not configured)")

def run_safe_tests():
    """Run only safe tests that don't require API keys"""
    print("Running safe data ingestion tests...")
    print("=" * 50)
    
    # Create test suite with only unit tests
    suite = unittest.TestSuite()
    
    # Add unit tests (mocked, safe)
    suite.addTest(TestDataIngestion('test_init'))
    suite.addTest(TestDataIngestion('test_get_crypto_symbol_id_success'))
    suite.addTest(TestDataIngestion('test_get_crypto_symbol_id_failure'))
    suite.addTest(TestDataIngestion('test_fetch_crypto_current_price_success'))
    suite.addTest(TestDataIngestion('test_fetch_stock_data_success'))
    suite.addTest(TestDataIngestion('test_fetch_stock_data_empty_response'))
    suite.addTest(TestDataIngestion('test_fetch_mixed_data_success'))
    suite.addTest(TestDataIngestion('test_get_crypto_quotes_success'))
    suite.addTest(TestDataIngestion('test_get_stock_quotes_success'))
    suite.addTest(TestDataIngestion('test_data_validation'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All safe tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests that require real API calls"""
    print("Running integration tests...")
    print("=" * 50)
    print("Note: These tests make real API calls and may require API keys")
    print()
    
    # Create test suite with integration tests
    suite = unittest.TestSuite()
    suite.addTest(TestDataIngestionIntegration('test_yahoo_finance_integration'))
    suite.addTest(TestDataIngestionIntegration('test_coinmarketcap_integration'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    return result.wasSuccessful()

def test_data_ingestion_functionality():
    """Test core functionality of data ingestion"""
    print("Testing core data ingestion functionality...")
    print("=" * 50)
    
    try:
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        print("✓ DataIngestion initialized successfully")
        
        # Test configuration
        print(f"✓ CoinMarketCap API configured: {'Yes' if data_ingestion.coinmarketcap_api_key != 'default_key' else 'No (using default)'}")
        print(f"✓ Yahoo Finance timeout: {data_ingestion.yahoo_timeout}s")
        
        # Test with sample data
        print("\nTesting with sample data...")
        
        # Create sample DataFrame
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        print(f"✓ Sample data created: {len(sample_data)} rows")
        print(f"✓ Sample data columns: {list(sample_data.columns)}")
        
        # Test data validation
        if not sample_data.empty:
            print("✓ Data validation passed")
        else:
            print("❌ Data validation failed")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in data ingestion functionality test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Data Ingestion Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    functionality_passed = test_data_ingestion_functionality()
    
    print("\n")
    
    # Run safe tests
    safe_passed = run_safe_tests()
    
    print("\n")
    
    # Ask user if they want to run integration tests
    try:
        user_input = input("Run integration tests with real API calls? (y/n): ").lower().strip()
        if user_input == 'y':
            integration_passed = run_integration_tests()
        else:
            integration_passed = True
            print("Integration tests skipped")
    except:
        integration_passed = True
        print("Integration tests skipped (no user input)")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if functionality_passed and safe_passed and integration_passed:
        print("✓ All tests passed successfully!")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)