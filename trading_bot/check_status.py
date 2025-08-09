import os
import time
import json
import pandas as pd
from datetime import datetime
import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_bot_status():
    """Check the current status of the trading bot."""
    try:
        # Initialize exchange
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
        })
        
        # Get account balance
        balance = exchange.fetch_balance()
        
        # Get open orders
        open_orders = exchange.fetch_open_orders()
        
        # Get recent trades
        recent_trades = exchange.fetch_my_trades(limit=10)
        
        # Get market data
        ticker = exchange.fetch_ticker('BTC/USD')
        
        # Check log file
        log_entries = []
        if os.path.exists('trading_bot.log'):
            with open('trading_bot.log', 'r') as f:
                log_entries = f.readlines()[-10:]  # Get last 10 log entries
        
        return {
            'timestamp': datetime.now().isoformat(),
            'balance': {k: v for k, v in balance['total'].items() if v > 0},
            'open_orders': [{
                'symbol': o['symbol'],
                'side': o['side'],
                'amount': o['amount'],
                'price': o['price'],
                'status': o['status']
            } for o in open_orders],
            'recent_trades': [{
                'symbol': t['symbol'],
                'side': t['side'],
                'amount': t['amount'],
                'price': t['price'],
                'cost': t['cost'],
                'datetime': t['datetime']
            } for t in recent_trades],
            'market': {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume']
            },
            'recent_logs': log_entries
        }
        
    except Exception as e:
        return {'error': str(e)}

def display_status():
    """Display the bot status in a human-readable format."""
    status = get_bot_status()
    
    if 'error' in status:
        print(f"Error getting status: {status['error']}")
        return
    
    print("\n" + "="*80)
    print(f"TRADING BOT STATUS - {status['timestamp']}")
    print("="*80)
    
    # Display balance
    print("\nBALANCE:")
    print("-"*40)
    for currency, amount in status['balance'].items():
        print(f"{currency}: {amount:.8f}")
    
    # Display open orders
    if status['open_orders']:
        print("\nOPEN ORDERS:")
        print("-"*40)
        for order in status['open_orders']:
            print(f"{order['symbol']} {order['side'].upper()}: {order['amount']} @ {order['price']}")
    else:
        print("\nNo open orders.")
    
    # Display recent trades
    if status['recent_trades']:
        print("\nRECENT TRADES:")
        print("-"*40)
        for trade in status['recent_trades']:
            print(f"{trade['datetime']} - {trade['symbol']} {trade['side'].upper()} "
                  f"{trade['amount']} @ {trade['price']} (${float(trade['cost']):.2f})")
    
    # Display market data
    print("\nMARKET DATA:")
    print("-"*40)
    m = status['market']
    print(f"{m['symbol']} - Last: {m['last']}, Bid: {m['bid']}, Ask: {m['ask']}, Volume: {m['volume']}")
    
    # Display recent logs
    if status['recent_logs']:
        print("\nRECENT LOGS:")
        print("-"*40)
        for log in status['recent_logs']:
            print(log.strip())
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        while True:
            display_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
