#!/usr/bin/env python3
"""Test script to verify Binance API connection."""
import os
import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.live')

def create_binanceus_session():
    """Create a requests session with forced IPv4 for Binance.US."""
    import requests
    from requests.adapters import HTTPAdapter
    import socket
    import urllib3
    
    # Monkey patch getaddrinfo to force IPv4
    original_getaddrinfo = socket.getaddrinfo
    
    def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    
    # Apply the monkey patch
    socket.getaddrinfo = getaddrinfo_ipv4
    
    # Create a session with custom settings
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = urllib3.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    
    # Create a custom adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    
    # Mount the custom adapter
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Set default headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    })
    
    # Set timeouts
    session.timeout = 30
    
    return session

# Initialize Binance.US exchange with custom session
exchange = ccxt.binanceus({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'warnOnFetchOpenOrdersWithoutSymbol': False,
    },
    'urls': {
        'api': {
            'public': 'https://api.binance.us/api/v3',
            'private': 'https://api.binance.us/api/v3',
        }
    },
    'session': create_binanceus_session()
})

try:
    # Test connectivity
    print("Testing Binance API connection...")
    
    # Test 1: Fetch server time (public endpoint, no authentication needed)
    server_time = exchange.fetch_time()
    print(f"✅ Server time: {exchange.iso8601(server_time)}")
    
    # Test 2: Fetch account balance (requires authentication)
    print("\nTesting authentication...")
    balance = exchange.fetch_balance()
    print("✅ Authentication successful")
    print("\nAccount Balances:")
    for currency, amount in balance['total'].items():
        if amount > 0:
            print(f"{currency}: {amount}")
    
    # Test 3: Check if we can fetch market data
    print("\nTesting market data...")
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"✅ Market data working. Current BTC/USDT price: {ticker['last']}")
    
    print("\n✅ All tests passed! The Binance API connection is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Verify your API key and secret in .env.live")
    print("2. Check if your IP is whitelisted in Binance")
    print("3. Ensure your account has the necessary permissions")
    print("4. Check if Binance API is experiencing any outages")
