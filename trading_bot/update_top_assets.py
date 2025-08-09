#!/usr/bin/env python3
"""
Script to update the .env file with top 25 assets by volume under $5.
"""
import ccxt
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import time

# Load environment variables
load_dotenv()

def get_top_assets() -> List[str]:
    """Fetch all assets under $5 with liquidity over $500,000."""
    try:
        # Initialize Kraken exchange
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000  # Increase timeout to 60 seconds
            }
        })
        
        print("🔍 Fetching all trading pairs from Kraken...")
        markets = exchange.load_markets()
        print(f"✅ Loaded {len(markets)} trading pairs")
        
        # Fetch tickers for all markets
        tickers = exchange.fetch_tickers()
        print(f"✅ Fetched {len(tickers)} tickers")
        
        # Filter and sort assets
        assets = []
        min_liquidity = 500000  # $500,000 minimum 24h volume
        
        for symbol, ticker in tickers.items():
            try:
                # Only consider USD pairs under $5 with sufficient liquidity
                if (symbol.endswith('/USD') and 
                    ticker.get('last') and 
                    ticker['last'] <= 5.0 and 
                    ticker.get('quoteVolume', 0) >= min_liquidity):
                    
                    assets.append({
                        'symbol': symbol,
                        'volume': float(ticker.get('quoteVolume', 0)),
                        'price': float(ticker.get('last', 0)),
                        'liquidity': float(ticker.get('quoteVolume', 0))  # 24h volume in quote currency (USD)
                    })
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)[:100]}...")
                continue
        
        # Sort by volume in descending order
        assets.sort(key=lambda x: x['volume'], reverse=True)
        
        # Log summary
        print(f"\n📊 Found {len(assets)} assets under $5 with >${min_liquidity:,.0f} 24h volume")
        if assets:
            print("\nTop 10 assets by volume:")
            for i, asset in enumerate(assets[:10], 1):
                print(f"{i:2d}. {asset['symbol']:12} | ${asset['price']:8.4f} | ${asset['volume']:12,.2f} 24h vol")
        
        return [asset['symbol'] for asset in assets]
    
    except Exception as e:
        print(f"Error fetching top assets: {e}")
        return []

def update_env_file(symbols: List[str]):
    """Update the .env file with new symbols."""
    env_file = '.env'
    
    # Read current .env file
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: .env file not found")
        return
    
    # Update KRAKEN_SYMBOLS line
    updated_lines = []
    symbols_str = ','.join(symbols)
    
    for line in lines:
        if line.startswith('KRAKEN_SYMBOLS='):
            updated_lines.append(f'KRAKEN_SYMBOLS={symbols_str}  # Auto-updated top {len(symbols)} assets by volume\n')
        else:
            updated_lines.append(line)
    
    # Write back to .env file
    try:
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated .env with {len(symbols)} trading pairs")
    except Exception as e:
        print(f"Error updating .env file: {e}")

def main():
    print("Fetching top 25 assets by volume under $5...")
    top_assets = get_top_assets()
    
    if not top_assets:
        print("No assets found. Using default list.")
        top_assets = [
            'BTC/USD', 'ETH/USD', 'XRP/USD', 'DOGE/USD', 'ADA/USD',
            'SOL/USD', 'MATIC/USD', 'SHIB/USD', 'DOT/USD', 'LTC/USD',
            'AVAX/USD', 'LINK/USD', 'ATOM/USD', 'UNI/USD', 'XLM/USD',
            'ALGO/USD', 'FIL/USD', 'HBAR/USD', 'VET/USD', 'THETA/USD',
            'XTZ/USD', 'XMR/USD', 'EOS/USD', 'AAVE/USD', 'CAKE/USD'
        ]
    
    print(f"Top {len(top_assets)} assets by volume under $5:")
    for i, symbol in enumerate(top_assets, 1):
        print(f"{i}. {symbol}")
    
    update_env_file(top_assets)
    print("\nConfiguration updated. Restarting bot...")
    
    # Restart the bot
    try:
        import subprocess
        import sys
        import os
        
        # Kill existing bot process
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        else:  # Unix/Linux/Mac
            subprocess.run(['pkill', '-f', 'kraken_scalper_fixed.py'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        
        # Start new bot process
        subprocess.Popen([sys.executable, 'kraken_scalper_fixed.py'],
                        cwd=os.getcwd(),
                        stdout=open('kraken_scalper.log', 'a'),
                        stderr=open('kraken_scalper.log', 'a'))
        
        print("Bot restarted successfully!")
        
    except Exception as e:
        print(f"Error restarting bot: {e}")
        print("Please restart the bot manually.")

if __name__ == "__main__":
    main()
