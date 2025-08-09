import os
import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.live')

def check_kraken_balance():
    try:
        kraken = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
        })
        
        print("\n=== KRAKEN BALANCE ===")
        balance = kraken.fetch_balance()
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"{currency}: {amount}")
        
        # Get tickers for non-zero balances to show USD value
        print("\n=== KRAKEN BALANCE (USD) ===")
        for currency, amount in balance['total'].items():
            if amount > 0 and currency != 'USD':
                try:
                    ticker = kraken.fetch_ticker(f"{currency}/USD")
                    usd_value = amount * ticker['last']
                    print(f"{currency}: {amount} (${usd_value:.2f})")
                except:
                    print(f"{currency}: {amount} (No USD pair)")
            elif currency == 'USD':
                print(f"USD: {amount}")
                
    except Exception as e:
        print(f"Error checking Kraken balance: {e}")

def check_binance_balance():
    try:
        binance = ccxt.binanceus({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            },
        })
        
        print("\n=== BINANCE BALANCE ===")
        balance = binance.fetch_balance()
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"{currency}: {amount}")
        
        # Get tickers for non-zero balances to show USD value
        print("\n=== BINANCE BALANCE (USDT) ===")
        for currency, amount in balance['total'].items():
            if amount > 0 and currency != 'USDT':
                try:
                    ticker = binance.fetch_ticker(f"{currency}/USDT")
                    usdt_value = amount * ticker['last']
                    print(f"{currency}: {amount} (${usdt_value:.2f})")
                except:
                    print(f"{currency}: {amount} (No USDT pair)")
            elif currency == 'USDT':
                print(f"USDT: {amount}")
                
    except Exception as e:
        print(f"Error checking Binance balance: {e}")

if __name__ == "__main__":
    check_kraken_balance()
    check_binance_balance()
