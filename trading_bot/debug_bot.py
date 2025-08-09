#!/usr/bin/env python3
"""Debug script for the Kraken trading bot."""
import os
import json
import time
import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_exchange_connection():
    """Check connection to Kraken exchange."""
    try:
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        
        # Test connection
        markets = exchange.load_markets()
        ticker = exchange.fetch_ticker('BTC/USD')
        
        return {
            'status': '✅ Connected to Kraken',
            'server_time': exchange.iso8601(exchange.milliseconds()),
            'markets_loaded': len(markets) > 0,
            'btc_price': ticker['last'] if ticker else 'N/A'
        }
    except Exception as e:
        return {
            'status': f'❌ Connection failed: {str(e)}',
            'error': str(e)
        }

def check_bot_files():
    """Check if all required bot files exist."""
    files = [
        'kraken_scalper_fixed.py',
        '.env',
        'kraken_scalper.log'
    ]
    
    results = {}
    for file in files:
        exists = os.path.exists(file)
        results[file] = '✅ Found' if exists else '❌ Missing'
        if exists:
            results[f'{file}_size'] = f"{os.path.getsize(file) / 1024:.1f} KB"
    
    return results

def check_logs():
    """Check the most recent log entries."""
    log_file = 'kraken_scalper.log'
    if not os.path.exists(log_file):
        return {'status': 'No log file found'}
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Get last 5 non-empty lines
        recent = [line.strip() for line in lines if line.strip()][-5:]
        
        return {
            'status': 'Log file found',
            'last_modified': time.ctime(os.path.getmtime(log_file)),
            'line_count': len(lines),
            'recent_entries': recent
        }
    except Exception as e:
        return {'status': f'Error reading log file: {str(e)}'}

def main():
    print("🔧 Kraken Trading Bot Debugger")
    print("=" * 50)
    
    # 1. Check exchange connection
    print("\n🔌 Exchange Connection:")
    exchange_status = check_exchange_connection()
    for key, value in exchange_status.items():
        print(f"   {key}: {value}")
    
    # 2. Check bot files
    print("\n📁 Bot Files:")
    files = check_bot_files()
    for file, status in files.items():
        if '_size' not in file:
            print(f"   {file}: {status} {files.get(f'{file}_size', '')}")
    
    # 3. Check logs
    print("\n📋 Log Status:")
    log_status = check_logs()
    for key, value in log_status.items():
        if key != 'recent_entries':
            print(f"   {key}: {value}")
    
    if 'recent_entries' in log_status and log_status['recent_entries']:
        print("\n   Recent Log Entries:")
        for entry in log_status['recent_entries']:
            print(f"   - {entry}")
    
    print("\n💡 Recommendations:")
    if 'error' in exchange_status:
        print(f"   - Fix exchange connection: {exchange_status['error']}")
    if 'kraken_scalper_fixed.py' not in files:
        print("   - Main bot file is missing. Please redownload or restore it.")
    if '.env' not in files:
        print("   - .env file is missing. Create one with your API credentials.")
    if 'kraken_scalper.log' not in files:
        print("   - No log file found. The bot may not have started properly.")
    
    print("\n✅ Debug check complete.")

if __name__ == "__main__":
    main()
