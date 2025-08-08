#!/usr/bin/env python3
"""
Divina Scalping Bot - Quick Start Guide

This script helps you start live trading with small amounts.
"""
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger('TradingStarter')

async def check_requirements():
    """Check if all required environment variables are set."""
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'TRADING_PAIR',
        'TRADE_AMOUNT',
        'LEVERAGE',
        'MAX_OPEN_TRADES'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.info("Please update your .env file with the required variables.")
        return False
    return True

async def start_bot():
    """Start the trading bot with the enhanced strategy."""
    from enhanced_scalper import EnhancedDivinaScalper, main
    
    logger.info("Starting Enhanced Divina Scalping Bot...")
    logger.info(f"Trading Pair: {os.getenv('TRADING_PAIR')}")
    logger.info(f"Trade Amount: ${float(os.getenv('TRADE_AMOUNT', '50'))}")
    logger.info(f"Leverage: {int(os.getenv('LEVERAGE', '2'))}x")
    logger.info(f"Max Open Trades: {int(os.getenv('MAX_OPEN_TRADES', '1'))}")
    
    try:
        await main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error in bot: {e}", exc_info=True)

async def start_monitoring():
    """Start the monitoring service."""
    from monitor_bot import TradingMonitor
    
    logger.info("Starting Trading Monitor...")
    monitor = TradingMonitor()
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Error in monitor: {e}", exc_info=True)

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv('.env.live')
    
    # Check requirements
    if not await check_requirements():
        sys.exit(1)
    
    # Start bot and monitor in parallel
    bot_task = asyncio.create_task(start_bot())
    monitor_task = asyncio.create_task(start_monitoring())
    
    try:
        await asyncio.gather(bot_task, monitor_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
        bot_task.cancel()
        monitor_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            bot_task,
            monitor_task,
            return_exceptions=True
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
