#!/usr/bin/env python3
"""
Test script to verify Kraken Pro API connection and basic functionality.
"""
import os
import ccxt
from dotenv import load_dotenv
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kraken_connection_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KrakenTest')

def test_kraken_connection():
    """Test connection to Kraken Pro API and basic functionality."""
    try:
        # Load environment variables
        load_dotenv('.env.live')
        
        # Initialize Kraken exchange
        kraken = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'createMarketBuyOrderRequiresPrice': False,  # Allow market orders without price
            },
        })
        
        logger.info("Successfully initialized Kraken exchange")
        
        # Test public API - fetch ticker
        ticker = kraken.fetch_ticker('SHIB/USD')
        logger.info(f"SHIB/USD Ticker: {ticker['last']} USD")
        
        # Test private API - fetch balance
        balance = kraken.fetch_balance()
        usd_balance = balance.get('USD', {}).get('free', 0)
        logger.info(f"Available USD Balance: {usd_balance} USD")
        
        # Test private API - fetch open orders
        open_orders = kraken.fetch_open_orders()
        logger.info(f"Number of open orders: {len(open_orders)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Kraken API: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"API Response: {e.response.text}")
        return False

if __name__ == "__main__":
    logger.info("Starting Kraken API connection test...")
    success = test_kraken_connection()
    if success:
        logger.info("✅ Kraken API test completed successfully")
    else:
        logger.error("❌ Kraken API test failed")
