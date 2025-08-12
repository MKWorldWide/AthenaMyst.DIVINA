"""
Test script for Oanda trading bot with multiple currency pairs.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.oanda_engine import OandaTradingEngine

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# File handler
file_handler = logging.FileHandler('oanda_multi_pair_test.log')
file_handler.setFormatter(log_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger('OandaMultiPairTest')

# Disable debug logging for oandapyV20 to reduce noise
logging.getLogger('oandapyV20').setLevel(logging.WARNING)

async def test_multi_pair_trading():
    """Test the Oanda trading bot with multiple currency pairs."""
    # Use the top 5 forex pairs for testing
    test_pairs = [
        'EUR_USD',  # Euro / US Dollar
        'USD_JPY',  # US Dollar / Japanese Yen
        'GBP_USD',  # British Pound / US Dollar
        'USD_CHF',  # US Dollar / Swiss Franc
        'AUD_USD',  # Australian Dollar / US Dollar
        'USD_CAD',  # US Dollar / Canadian Dollar
        'NZD_USD',  # New Zealand Dollar / US Dollar
        'EUR_GBP',  # Euro / British Pound
        'EUR_JPY',  # Euro / Japanese Yen
        'GBP_JPY'   # British Pound / Japanese Yen
    ][:5]  # Limit to first 5 pairs for initial testing
    
    logger.info("=" * 80)
    logger.info("OANDA MULTI-PAIR TRADING TEST")
    logger.info("=" * 80)
    logger.info(f"Starting test at {datetime.now().isoformat()}")
    logger.info(f"Test pairs: {', '.join(test_pairs)}")
    logger.info("-" * 80)
    
    try:
        logger.info("Initializing OandaTradingEngine...")
        
        try:
            # Initialize the trading engine
            bot = OandaTradingEngine(
                config_file='.env.oanda',
                dry_run=True,  # Run in dry-run mode (no real trades)
                trading_pairs=test_pairs  # Pass pairs directly to constructor
            )
            logger.info("OandaTradingEngine initialized successfully")
            
            # Verify initialization
            if not hasattr(bot, 'client') or bot.client is None:
                raise RuntimeError("OANDA client not properly initialized")
                
            logger.info(f"Account ID: {getattr(bot, 'account_id', 'Not set')}")
            logger.info(f"Account Type: {getattr(bot, 'account_type', 'Not set')}")
            logger.info(f"Dry Run Mode: {getattr(bot, 'dry_run', 'Not set')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OandaTradingEngine: {e}", exc_info=True)
            logger.error("Please check your .env.oanda file and OANDA credentials")
            return
        
        logger.info(f"Testing Oanda trading bot with {len(test_pairs)} currency pairs")
        logger.info(f"Pairs: {', '.join(test_pairs)}")
        
        # Test price fetching for each pair
        logger.info("\n=== Testing price fetching ===")
        for pair in test_pairs:
            try:
                price = await bot.get_current_price(pair)
                if price is not None:
                    logger.info(f"{pair}: {price:.5f}")
                else:
                    logger.warning(f"Failed to get price for {pair}")
            except Exception as e:
                logger.error(f"Error getting price for {pair}: {e}")
        
        # Test signal generation for each pair
        logger.info("\n=== Testing signal generation ===")
        for pair in test_pairs:
            try:
                signal = await bot.generate_signal(pair)
                indicators = bot.pair_states[pair].get('indicators', {})
                price = indicators.get('price', 'N/A')
                sma20 = indicators.get('sma20', 'N/A')
                sma50 = indicators.get('sma50', 'N/A')
                rsi = indicators.get('rsi', 'N/A')
                
                logger.info(
                    f"{pair}: {signal} | "
                    f"Price: {price:.5f} | "
                    f"SMA20: {sma20:.5f} | "
                    f"SMA50: {sma50:.5f} | "
                    f"RSI: {rsi:.2f}"
                )
            except Exception as e:
                logger.error(f"Error generating signal for {pair}: {e}")
        
        # Test the main trading loop with a short duration
        logger.info("\n=== Testing trading loop (30 seconds) ===")
        logger.info("Press Ctrl+C to stop the test")
        
        async def monitor_loop():
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")
                
                # Check account status
                try:
                    account = await bot.get_account_summary()
                    if account:
                        logger.info(
                            f"Account: {account.get('alias', 'N/A')} | "
                            f"Balance: {account.get('balance', 'N/A')} {account.get('currency', 'USD')} | "
                            f"Margin Available: {account.get('marginAvailable', 'N/A')}"
                        )
                except Exception as e:
                    logger.error(f"Error getting account summary: {e}")
                
                # Process each pair
                for pair in test_pairs:
                    try:
                        # Get current price
                        price = await bot.get_current_price(pair)
                        if price is None:
                            logger.warning(f"Skipping {pair} - No price data")
                            continue
                        
                        # Generate signal
                        signal = await bot.generate_signal(pair)
                        indicators = bot.pair_states[pair].get('indicators', {})
                        
                        logger.info(
                            f"{pair}: {signal} | "
                            f"Price: {price:.5f} | "
                            f"SMA20: {indicators.get('sma20', 'N/A'):.5f} | "
                            f"SMA50: {indicators.get('sma50', 'N/A'):.5f} | "
                            f"RSI: {indicators.get('rsi', 'N/A'):.2f}"
                        )
                        
                        # Simulate trade execution (in dry-run mode)
                        if signal in ['BUY', 'SELL']:
                            logger.info(f"  Would execute {signal} order for {pair} at {price:.5f}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {pair}: {e}")
                
                # Wait before next iteration
                await asyncio.sleep(10)  # 10 seconds between iterations
        
        # Run the monitoring loop for 30 seconds
        await asyncio.wait_for(monitor_loop(), timeout=30)
        
    except asyncio.CancelledError:
        logger.info("Test cancelled by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        logger.info("Test completed")

if __name__ == "__main__":
    try:
        asyncio.run(test_multi_pair_trading())
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
