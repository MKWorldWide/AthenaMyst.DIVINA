#!/usr/bin/env python3
"""
Test script to verify live trading functionality with Kraken.
"""
import ccxt
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def place_test_trade():
    """Place a small test trade on Kraken."""
    try:
        # Initialize Kraken exchange
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'trading_agreement': 'agree',
            },
        })
        
        # Test connection
        exchange.load_markets()
        print("✅ Successfully connected to Kraken")
        
        # Get account balance
        balance = exchange.fetch_balance()
        usd_balance = balance.get('USD', {}).get('free', 0)
        print(f"💰 Available USD Balance: ${usd_balance:.2f}")
        
        # Find a suitable trading pair (USDT/USD is usually very liquid)
        symbol = 'USDT/USD'
        print(f"\n🔍 Testing with {symbol}")
        
        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"📈 Current {symbol} price: ${current_price:.6f}")
        
        # Calculate test order size ($10 worth to meet minimums)
        amount = 10.0 / current_price
        
        # Place test buy order
        print(f"\n🛒 Placing test buy order for {amount:.6f} {symbol} (≈$1.00)")
        order = exchange.create_market_buy_order(symbol, amount)
        print(f"✅ Buy order placed: {order['id']}")
        
        # Wait a moment
        time.sleep(2)
        
        # Get order status
        order_status = exchange.fetch_order(order['id'], symbol)
        print(f"\n📊 Order status:")
        print(f"   Status: {order_status['status']}")
        print(f"   Filled: {order_status['filled']}")
        print(f"   Cost: ${float(order_status['cost'] or 0):.2f}")
        
        # Place sell order to close position
        if float(order_status['filled'] or 0) > 0:
            print("\n🔄 Placing sell order to close position...")
            sell_order = exchange.create_market_sell_order(symbol, float(order_status['filled']))
            print(f"✅ Sell order placed: {sell_order['id']}")
        
        print("\n🎉 Test trade completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test trade: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    print("🚀 Starting Kraken Test Trade")
    print("=" * 50)
    place_test_trade()
