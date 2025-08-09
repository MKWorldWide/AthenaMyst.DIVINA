#!/usr/bin/env python3
"""
Kraken Live Trading Test Script

This script tests live trading with Kraken Pro using a small amount.
It will place a small market order and then cancel it immediately.
"""
import os
import ccxt
from dotenv import load_dotenv
import logging
from decimal import Decimal, ROUND_DOWN
import time
import sys

# Add the parent directory to the path to allow importing serafina_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.serafina_integration import report_trade_to_serafina

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kraken_live_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KrakenLiveTest')

def get_kraken_exchange():
    """Initialize and return a configured Kraken exchange instance."""
    return ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY'),
        'secret': os.getenv('KRAKEN_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'createMarketBuyOrderRequiresPrice': False,
        },
    })

def get_market_info(exchange, symbol):
    """Get market information for the given symbol."""
    market = exchange.markets[symbol]
    return {
        'symbol': symbol,
        'base': market['base'],
        'quote': market['quote'],
        'precision': market['precision'],
        'limits': market['limits'],
        'active': market['active']
    }

def get_balance(exchange, currency):
    """Get available balance for the specified currency."""
    balance = exchange.fetch_balance()
    return {
        'free': balance.get(currency, {}).get('free', 0),
        'used': balance.get(currency, {}).get('used', 0),
        'total': balance.get(currency, {}).get('total', 0)
    }

def format_amount(amount, precision):
    """Format amount to the required precision."""
    return float(Decimal(str(amount)).quantize(
        Decimal(str(precision)),
        rounding=ROUND_DOWN
    ))

def test_live_trade():
    """Test live trading with a small amount."""
    try:
        # Load environment variables
        load_dotenv('.env.live')
        
        # Initialize exchange
        kraken = get_kraken_exchange()
        logger.info("Successfully connected to Kraken")
        
        # Get trading pair from environment
        symbol = os.getenv('TRADING_PAIR', 'SHIB/USD')
        logger.info(f"Using trading pair: {symbol}")
        
        # Get market info
        market_info = get_market_info(kraken, symbol)
        logger.info(f"Market info: {market_info}")
        
        # Get current balance
        quote_currency = market_info['quote']
        balance = get_balance(kraken, quote_currency)
        logger.info(f"{quote_currency} Balance: {balance}")
        
        # Calculate order amount (small test amount)
        trade_amount = float(os.getenv('TRADE_AMOUNT', '10'))  # Default to $10
        min_cost = market_info['limits']['cost']['min']
        
        if trade_amount < min_cost:
            logger.warning(f"Trade amount ${trade_amount} is below minimum cost of ${min_cost}")
            trade_amount = min_cost * 1.1  # Add 10% buffer
            logger.info(f"Adjusted trade amount to: ${trade_amount:.2f}")
        
        # Get current price
        ticker = kraken.fetch_ticker(symbol)
        current_price = ticker['last']
        logger.info(f"Current {symbol} price: ${current_price}")
        
        # Calculate amount in base currency
        amount = trade_amount / current_price
        amount = format_amount(amount, 10 ** -market_info['precision']['amount'])
        
        logger.info(f"Preparing to place order: {amount} {market_info['base']} at market price")
        
        # Place a limit order (safer for testing)
        order = kraken.create_order(
            symbol=symbol,
            type='limit',
            side='buy',
            amount=amount,
            price=current_price * 0.9,  # 10% below market to avoid immediate fill
            params={'validate': True}  # Validate only, don't place real order
        )
        
        # Place real market order for a small amount
        logger.info("Placing real market order for a small amount...")
        
        # Reduce the amount to a very small value for testing (1000 SHIB ~= $0.01)
        test_amount = 1000  # 1000 SHIB for testing
        logger.info(f"Placing order for {test_amount} SHIB (~${test_amount * current_price:.4f})")
        
        try:
            # Place buy order
            order = kraken.create_order(
                symbol=symbol,
                type='market',
                side='buy',
                amount=test_amount
            )
            logger.info(f"✅ Buy order placed: {order['id']}")
            
            # Monitor order status
            time.sleep(5)  # Wait for order to fill
            order_status = kraken.fetch_order(order['id'], symbol)
            logger.info(f"Order status: {order_status['status']}")
            
            if order_status['status'] == 'closed':
                filled_amount = float(order_status['filled'])
                avg_price = float(order_status['average'])
                cost = filled_amount * avg_price
                logger.info(f"✅ Order filled. {filled_amount} {market_info['base']} at {avg_price} {quote_currency} (Total: ${cost:.4f})")
                
                # Small delay before selling
                logger.info("Waiting 10 seconds before placing sell order...")
                time.sleep(10)
                
                # Place sell order to close position
                sell_order = kraken.create_order(
                    symbol=symbol,
                    type='market',
                    side='sell',
                    amount=filled_amount
                )
                logger.info(f"✅ Sell order placed: {sell_order['id']}")
                
                # Verify sell order
                time.sleep(5)
                sell_status = kraken.fetch_order(sell_order['id'], symbol)
                if sell_status['status'] == 'closed':
                    sell_price = float(sell_status['average'])
                    profit_loss = (sell_price - avg_price) * filled_amount
                    profit_pct = ((sell_price/avg_price)-1)*100
                    
                    logger.info(f"✅ Position closed at {sell_price} {quote_currency}")
                    logger.info(f"💰 P&L: ${profit_loss:.4f} ({profit_pct:.2f}%)")
                    
                    # Report trades to Serafina
                    try:
                        # Report buy
                        buy_success = report_trade_to_serafina(
                            symbol=symbol,
                            side='buy',
                            amount=filled_amount,
                            price=avg_price,
                            order_id=order['id'],
                            exchange='kraken',
                            notes='Test trade - Buy'
                        )
                        if buy_success:
                            logger.info("✅ Buy trade reported to Serafina")
                        
                        # Report sell
                        sell_success = report_trade_to_serafina(
                            symbol=symbol,
                            side='sell',
                            amount=filled_amount,
                            price=sell_price,
                            order_id=sell_order['id'],
                            exchange='kraken',
                            notes=f'Test trade - Sell | P&L: ${profit_loss:.4f} ({profit_pct:.2f}%)'
                        )
                        if sell_success:
                            logger.info("✅ Sell trade reported to Serafina")
                            
                    except Exception as e:
                        logger.error(f"Error reporting to Serafina: {str(e)}")
                else:
                    logger.warning(f"Sell order status: {sell_status['status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during live trading: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"API Response: {e.response.text}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_live_trade: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"API Response: {e.response.text}")
        return False

if __name__ == "__main__":
    logger.info("Starting Kraken Live Trading Test...")
    logger.info("This is a TEST - No real orders will be placed unless you uncomment the code.")
    
    success = test_live_trade()
    if success:
        logger.info("✅ Kraken live trading test completed successfully")
    else:
        logger.error("❌ Kraken live trading test failed")
