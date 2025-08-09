import os
import ccxt
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BalanceChecker')

def setup_exchange(exchange_name, api_key, api_secret):
    """Initialize and return the exchange with API credentials."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
        },
        'verbose': False
    })
    
    # Force IPv4 for Binance.US
    if exchange_name.lower() == 'binanceus':
        exchange.session.verify = True
        exchange.session.trust_env = False
        exchange.session.proxies = {}
        
    return exchange

def get_balance(exchange, exchange_name):
    """Fetch and display the account balance for the given exchange."""
    try:
        logger.info(f"Fetching {exchange_name} balance...")
        balance = exchange.fetch_balance()
        
        # Filter out zero balances and format the output
        non_zero_balances = {}
        for currency, amount in balance['total'].items():
            if amount > 0:
                non_zero_balances[currency] = amount
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{exchange_name.upper()} BALANCE")
        logger.info(f"{'='*50}")
        
        total_btc = 0
        total_usd = 0
        
        # Get tickers for all non-zero balances to calculate USD value
        if non_zero_balances:
            symbols = [f"{currency}/USDT" for currency in non_zero_balances.keys() if currency != 'USDT']
            if symbols:
                try:
                    tickers = exchange.fetch_tickers(symbols)
                except Exception as e:
                    logger.warning(f"Could not fetch all tickers: {e}")
                    tickers = {}
            else:
                tickers = {}
                
            # Display balances with USD values
            for currency, amount in non_zero_balances.items():
                usd_value = 0
                btc_value = 0
                
                if currency == 'USDT':
                    usd_value = amount
                    btc_ticker = f"BTC/USDT"
                    if btc_ticker in tickers:
                        btc_price = tickers[btc_ticker]['last']
                        btc_value = amount / btc_price if btc_price > 0 else 0
                else:
                    symbol = f"{currency}/USDT"
                    if symbol in tickers and tickers[symbol]['last'] is not None:
                        usd_value = amount * tickers[symbol]['last']
                    
                    btc_symbol = f"{currency}/BTC"
                    if btc_symbol in tickers and tickers[btc_symbol]['last'] is not None:
                        btc_value = amount * tickers[btc_symbol]['last']
                    elif 'BTC/USDT' in tickers and tickers['BTC/USDT']['last'] is not None and usd_value > 0:
                        btc_price = tickers['BTC/USDT']['last']
                        btc_value = usd_value / btc_price
                
                total_usd += usd_value
                total_btc += btc_value
                
                logger.info(f"{currency}: {amount:.8f} (${usd_value:.2f}, {btc_value:.8f} BTC)")
        
        logger.info(f"{'='*50}")
        logger.info(f"TOTAL: ${total_usd:.2f} ({total_btc:.8f} BTC)")
        logger.info(f"{'='*50}\n")
        
        return {
            'exchange': exchange_name,
            'total_usd': total_usd,
            'total_btc': total_btc,
            'balances': non_zero_balances
        }
        
    except Exception as e:
        logger.error(f"Error fetching {exchange_name} balance: {e}")
        return None

def main():
    # Load environment variables
    load_dotenv('.env.live')
    
    # Get API credentials
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_api_secret = os.getenv('BINANCE_API_SECRET')
    
    kraken_api_key = os.getenv('KRAKEN_API_KEY')
    kraken_api_secret = os.getenv('KRAKEN_API_SECRET')
    
    results = {}
    
    # Check Binance balance if credentials are available
    if binance_api_key and binance_api_secret:
        try:
            binance = setup_exchange('binanceus', binance_api_key, binance_api_secret)
            results['binance'] = get_balance(binance, 'Binance.US')
        except Exception as e:
            logger.error(f"Error initializing Binance: {e}")
    
    # Check Kraken balance if credentials are available
    if kraken_api_key and kraken_api_secret:
        try:
            kraken = setup_exchange('kraken', kraken_api_key, kraken_api_secret)
            results['kraken'] = get_balance(kraken, 'Kraken')
        except Exception as e:
            logger.error(f"Error initializing Kraken: {e}")
    
    return results

if __name__ == "__main__":
    main()
