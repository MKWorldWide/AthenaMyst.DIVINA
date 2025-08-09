import os
import ccxt
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BinanceTest')

def test_binance_connection():
    """Test connection to Binance US with detailed error handling."""
    try:
        # Load environment variables
        load_dotenv('.env.live')
        
        # Get API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("API key or secret not found in .env.live")
            return False
            
        logger.info("Initializing Binance.US exchange...")
        
        # Initialize exchange with explicit parameters
        exchange = ccxt.binanceus({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # Increase recvWindow to 60 seconds
            },
            'timeout': 30000,  # 30 second timeout
        })
        
        # Test connectivity
        logger.info("Testing connectivity...")
        exchange.fetch_status()
        
        # Test API key permissions
        logger.info("Testing API key permissions...")
        try:
            balance = exchange.fetch_balance()
            logger.info("Successfully fetched balance!")
            
            # Print non-zero balances
            logger.info("\n=== ACCOUNT BALANCE ===")
            total_balance = 0
            for currency, amount in balance['total'].items():
                if amount > 0:
                    logger.info(f"{currency}: {amount}")
                    
                    # Try to get USD value
                    if currency != 'USDT':
                        try:
                            ticker = exchange.fetch_ticker(f"{currency}/USDT")
                            usd_value = amount * ticker['last']
                            logger.info(f"  ${usd_value:.2f}")
                            total_balance += usd_value
                        except Exception as e:
                            logger.warning(f"  Could not fetch price for {currency}/USDT: {e}")
                    else:
                        total_balance += amount
                        
            logger.info(f"\nTOTAL BALANCE: ${total_balance:.2f}")
            
            return True
            
        except ccxt.ExchangeError as e:
            logger.error(f"API key error: {e}")
            if "API-key format invalid" in str(e):
                logger.error("The API key format is invalid. Please check your API key.")
            elif "Signature for this request is not valid" in str(e):
                logger.error("The API secret is invalid. Please check your API secret.")
            elif "IP" in str(e) and "not allowed" in str(e):
                logger.error("Your IP address is not whitelisted. Please add your IP to the API restrictions.")
            elif "This action is disabled on this account" in str(e):
                logger.error("Trading is disabled for this API key. Please enable trading in your API settings.")
            else:
                logger.error(f"Unhandled exchange error: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("\n=== BINANCE.US CONNECTION TEST ===\n")
    success = test_binance_connection()
    
    if success:
        print("\n✅ Binance.US connection successful!")
    else:
        print("\n❌ Binance.US connection failed. Please check the error messages above.")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Verify your API key and secret are correct")
        print("2. Check if your IP address is whitelisted in Binance.US API settings")
        print("3. Ensure 'Enable Trading' is enabled for the API key")
        print("4. Make sure the API key has not expired")
        print("5. Check if your account has 2FA enabled (you may need to disable it for API access)")
