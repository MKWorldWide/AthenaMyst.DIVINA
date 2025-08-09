#!/usr/bin/env python3
"""
Test script for the crypto trading engine.

This script tests the core functionality of the crypto trading engine
without making any real trades or API calls to exchanges.
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ccxt_engine import (
    discover_symbols,
    account_quote_balance,
    position_size,
    breakout_signal
)

class TestCryptoEngine(unittest.TestCase):
    """Test cases for the crypto trading engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock exchange with test data
        self.exchange = MagicMock()
        self.exchange.markets = {
            'BTC/USDT': {
                'active': True,
                'base': 'BTC',
                'quote': 'USDT',
                'precision': {'amount': 6, 'price': 2},
                'limits': {
                    'amount': {'min': 0.0001, 'max': 1000},
                    'cost': {'min': 10, 'max': None},
                    'price': {'min': 0.01, 'max': 1000000}
                }
            },
            'ETH/USDT': {
                'active': True,
                'base': 'ETH',
                'quote': 'USDT',
                'precision': {'amount': 4, 'price': 2},
                'limits': {
                    'amount': {'min': 0.001, 'max': 10000},
                    'cost': {'min': 10, 'max': None},
                    'price': {'min': 0.01, 'max': 100000}
                }
            },
            'SHIB/USDT': {
                'active': True,
                'base': 'SHIB',
                'quote': 'USDT',
                'precision': {'amount': 0, 'price': 8},
                'limits': {
                    'amount': {'min': 1000, 'max': 1000000000},
                    'cost': {'min': 10, 'max': None},
                    'price': {'min': 0.00000001, 'max': 0.01}
                }
            }
        }
        
        # Mock amount_to_precision method
        def mock_amount_to_precision(symbol, amount):
            market = self.exchange.markets[symbol]
            precision = market['precision']['amount']
            if precision == 0:
                return int(amount)
            return round(amount, precision)
            
        self.exchange.amount_to_precision = mock_amount_to_precision
        
        # Mock ticker data
        self.exchange.fetch_tickers.return_value = {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'last': 45000.0,
                'quoteVolume': 5000000.0,
                'baseVolume': 111.11
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'last': 3000.0,
                'quoteVolume': 2000000.0,
                'baseVolume': 666.66
            },
            'SHIB/USDT': {
                'symbol': 'SHIB/USDT',
                'last': 0.00003,
                'quoteVolume': 1000000.0,
                'baseVolume': 33333333333.33
            }
        }
        
        # Mock balance
        self.exchange.fetch_balance.return_value = {
            'free': {'USDT': 1000.0, 'BTC': 0.5, 'ETH': 5.0, 'SHIB': 0.0},
            'total': {'USDT': 1000.0, 'BTC': 0.5, 'ETH': 5.0, 'SHIB': 0.0}
        }
        
        # Mock OHLCV data
        self.ohlcv = [
            [int(time.time()*1000)-i*300000,  # Timestamp (5m intervals)
             1000.0,  # Open
             1010.0,  # High
             990.0,   # Low
             1005.0,  # Close
             100.0]   # Volume
            for i in range(50, -1, -1)  # Last 50 candles
        ]
        
        # Add a breakout to the most recent candle
        self.ohlcv[-1][2] = 1020.0  # New high
        self.ohlcv[-1][4] = 1018.0  # Close near high
        
        self.exchange.fetch_ohlcv.return_value = self.ohlcv
    
    def test_discover_symbols(self):
        """Test symbol discovery with price and volume filters."""
        # Test with high price filter (should exclude BTC/USDT and ETH/USDT)
        symbols = discover_symbols(self.exchange, max_price=1.0, min_vol_usd=500000)
        self.assertIn('SHIB/USDT', symbols)
        self.assertNotIn('BTC/USDT', symbols)
        self.assertNotIn('ETH/USDT', symbols)
        
        # Test with lower volume filter (should include all symbols)
        symbols = discover_symbols(self.exchange, max_price=50000.0, min_vol_usd=1000)
        self.assertIn('BTC/USDT', symbols)
        self.assertIn('ETH/USDT', symbols)
        self.assertIn('SHIB/USDT', symbols)
    
    def test_account_balance(self):
        """Test account balance retrieval."""
        balance = account_quote_balance(self.exchange, 'USDT')
        self.assertEqual(balance, 1000.0)
        
        # Test with missing balance (should return 0)
        balance = account_quote_balance(self.exchange, 'XRP')
        self.assertEqual(balance, 0.0)
    
    def test_position_size(self):
        """Test position size calculation."""
        # Test with BTC/USDT
        # With price=45000 and risk_amount=100, raw amount is 100/45000 = 0.002222...
        # The exchange has min_cost=10, but 100 * 45000 = 4500000 which is > 10, so no adjustment
        # The min_amount is 0.0001, which is smaller than 0.002222, so no adjustment
        # The final amount should be the raw amount rounded to the exchange's precision (6 decimal places)
        size = position_size(self.exchange, 'BTC/USDT', 100.0, 45000.0)
        expected_size = 0.002222  # 100/45000 rounded to 6 decimal places
        self.assertAlmostEqual(size, expected_size, places=8)
        
        # Test with SHIB/USDT (minimum cost check)
        # With price=0.00003 and risk_amount=5, raw amount is 5/0.00003 = 166,666.666...
        # The exchange has min_cost=10, but 5 * 0.00003 = 0.00015 which is < 10, so we adjust
        # We would need at least 10 USDT worth, so (10 * 1.01) / 0.00003 = 336,666.666...
        # However, this would exceed our risk amount (5 USDT), so we cap at 5/0.00003 = 166,666.666...
        # The min_amount is 1000, which is smaller than 166,666.666..., so no adjustment needed
        # Finally, we round to the exchange's precision (0 decimal places for SHIB)
        # Note: The implementation uses amount_to_precision which may use different rounding than Python's round()
        
        size = position_size(self.exchange, 'SHIB/USDT', 5.0, 0.00003)
        expected_shib = 166666  # 5/0.00003 ≈ 166,666.666... which gets floored to 166,666
        self.assertEqual(size, expected_shib)
    
    def test_breakout_signal(self):
        """Test breakout signal generation."""
        # Test with our mock data (should detect breakout)
        signal, price = breakout_signal(self.ohlcv, lookback=20)
        self.assertEqual(signal, 'buy')
        self.assertEqual(price, 1018.0)
        
        # Test with no breakout (should return 'hold')
        no_breakout = [candle.copy() for candle in self.ohlcv]
        no_breakout[-1][2] = 1009.0  # High below previous high
        no_breakout[-1][4] = 1007.0  # Close below previous high
        
        signal, _ = breakout_signal(no_breakout, lookback=20)
        self.assertEqual(signal, 'hold')

if __name__ == '__main__':
    unittest.main()
