#!/usr/bin/env python3
"""
Setup script for the 24/7 Crypto Trading Bot.

This script helps to:
1. Install required Python packages
2. Create a .env file with template configuration
3. Verify exchange connections
"""

import os
import sys
import subprocess
from getpass import getpass

def install_requirements():
    """Install required Python packages."""
    print("\n=== Installing required packages ===")
    requirements = [
        'ccxt>=3.0.0',
        'python-dotenv>=0.19.0',
        'requests>=2.26.0',
        'pytest>=6.2.5',
        'pytest-mock>=3.6.1'
    ]
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
        print("✅ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def create_env_file():
    """Create or update the .env file with user input."""
    env_path = '.env.crypto'
    
    # Check if .env file already exists
    if os.path.exists(env_path):
        print("\n⚠️  Found existing .env.crypto file. Creating a backup...")
        try:
            with open(env_path, 'r') as f:
                existing_content = f.read()
            with open(f'{env_path}.bak', 'w') as f:
                f.write(existing_content)
            print(f"✅ Created backup at {env_path}.bak")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
    
    print("\n=== Crypto Trading Bot Configuration ===")
    print("\nEnter your exchange API credentials (leave blank to skip):")
    
    # Binance.US API
    print("\n🔑 Binance.US API (optional):")
    binance_key = input("  API Key: ").strip()
    binance_secret = getpass("  API Secret: ").strip()
    
    # Kraken API
    print("\n🔑 Kraken API (optional):")
    kraken_key = input("  API Key: ").strip()
    kraken_secret = getpass("  API Secret: ").strip()
    
    # Trading parameters
    print("\n⚙️  Trading Parameters:")
    risk_per_trade = input(f"  Risk per trade (default: 0.01 for 1%): ").strip() or "0.01"
    max_open_trades = input("  Max open trades (default: 3): ").strip() or "3"
    timeframe = input("  Timeframe (default: 5m): ").strip() or "5m"
    
    # Generate .env content
    env_content = f"""# Crypto Trading Bot Configuration
# Exchange API Keys (keep these secure!)
BINANCEUS_API_KEY={'YOUR_BINANCEUS_API_KEY' if not binance_key else binance_key}
BINANCEUS_API_SECRET={'YOUR_BINANCEUS_API_SECRET' if not binance_secret else '***MASKED***'}
KRAKEN_API_KEY={'YOUR_KRAKEN_API_KEY' if not kraken_key else kraken_key}
KRAKEN_API_SECRET={'YOUR_KRAKEN_API_SECRET' if not kraken_secret else '***MASKED***'}

# Trading Parameters
RISK_PER_TRADE={risk_per_trade}  # Risk per trade as a fraction of account balance
MAX_OPEN_TRADES={max_open_trades}  # Maximum number of concurrent trades
TIMEFRAME={timeframe}  # Candle timeframe (e.g., 5m, 15m, 1h)

# Coin Discovery
MAX_PRICE=5.0  # Maximum price per coin in USD
MIN_VOL_USD=500000  # Minimum 24h trading volume in USD

# Risk Management
TP_PCT=0.004  # Take profit percentage (0.4%)
SL_PCT=0.003  # Stop loss percentage (0.3%)
COOLDOWN_MIN=20  # Cooldown between trades on the same pair (minutes)
"""
    
    # Write to .env file
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        # If secrets were provided, update them in the file
        if binance_secret or kraken_secret:
            with open(env_path, 'r') as f:
                content = f.read()
            
            if binance_secret:
                content = content.replace('***MASKED***', binance_secret)
            if kraken_secret:
                content = content.replace('***MASKED***', kraken_secret)
            
            with open(env_path, 'w') as f:
                f.write(content)
        
        print(f"\n✅ Configuration saved to {env_path}")
        
        # Create a symlink to .env for convenience
        if not os.path.exists('.env'):
            try:
                if os.name == 'nt':  # Windows
                    import ctypes
                    ctypes.windll.kernel32.CreateSymbolicLinkW('.env', env_path, 0)
                else:  # Unix-like
                    os.symlink(env_path, '.env')
                print(f"✅ Created symlink from .env to {env_path}")
            except Exception as e:
                print(f"⚠️  Could not create .env symlink: {e}")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        sys.exit(1)

def verify_exchanges():
    """Verify exchange connections with provided API keys."""
    from dotenv import load_dotenv
    import ccxt
    
    load_dotenv('.env.crypto')
    
    print("\n=== Verifying Exchange Connections ===")
    
    # Test Binance.US connection
    if os.getenv('BINANCEUS_API_KEY'):
        try:
            print("\n🔍 Testing Binance.US connection...")
            exchange = ccxt.binanceus({
                'apiKey': os.getenv('BINANCEUS_API_KEY'),
                'secret': os.getenv('BINANCEUS_API_SECRET'),
                'enableRateLimit': True
            })
            exchange.check_required_credentials()
            print("✅ Binance.US connection successful!")
            
            # Test balance fetch
            try:
                balance = exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                print(f"   Available USDT: {usdt_balance:.2f}")
            except Exception as e:
                print(f"⚠️  Could not fetch balance: {e}")
                
        except Exception as e:
            print(f"❌ Binance.US connection failed: {e}")
    else:
        print("\nℹ️  Skipping Binance.US (no API key provided)")
    
    # Test Kraken connection
    if os.getenv('KRAKEN_API_KEY'):
        try:
            print("\n🔍 Testing Kraken connection...")
            exchange = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_API_SECRET'),
                'enableRateLimit': True
            })
            exchange.check_required_credentials()
            print("✅ Kraken connection successful!")
            
            # Test balance fetch
            try:
                balance = exchange.fetch_balance()
                usd_balance = balance.get('USD', {}).get('free', 0)
                print(f"   Available USD: {usd_balance:.2f}")
            except Exception as e:
                print(f"⚠️  Could not fetch balance: {e}")
                
        except Exception as e:
            print(f"❌ Kraken connection failed: {e}")
    else:
        print("\nℹ️  Skipping Kraken (no API key provided)")

def main():
    """Main setup function."""
    print("""
╔══════════════════════════════════════════╗
║      24/7 Crypto Trading Bot Setup       ║
╚══════════════════════════════════════════╝
""")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run setup steps
    install_requirements()
    create_env_file()
    
    # Only verify exchanges if user wants to
    verify = input("\nWould you like to verify exchange connections now? (y/n): ").strip().lower()
    if verify == 'y':
        verify_exchanges()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Review the configuration in .env.crypto")
    print("2. Run tests: python -m pytest test_crypto_engine.py -v")
    print("3. Start the bot: python run_crypto.py")
    print("\nFor help: python run_crypto.py --help")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(0)
