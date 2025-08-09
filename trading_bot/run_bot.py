#!/usr/bin/env python3
"""
Main entry point for the crypto trading bot.

Usage:
    python run_bot.py --exchange kraken
    python run_bot.py --exchange binanceus
"""
import os
import sys
import time
import logging
import argparse
import threading
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.trading_loop import run_kraken, run_binanceus

def setup_logging() -> None:
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "trading_bot.log"),
                             encoding='utf-8')
        ]
    )
    
    # Suppress noisy loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def check_environment() -> bool:
    """Check that required environment variables are set."""
    required_vars = [
        "KRAKEN_API_KEY", "KRAKEN_API_SECRET",
        "BINANCE_API_KEY", "BINANCE_API_SECRET"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Please check your .env.live file")
        return False
    
    return True

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--exchange", "-e", 
                       choices=["kraken", "binanceus"],
                       help="Exchange to trade on")
    parser.add_argument("--all", action="store_true",
                       help="Run all configured exchanges in separate threads")
    
    args = parser.parse_args()
    
    if not args.exchange and not args.all:
        parser.print_help()
        print("\nError: Either --exchange or --all must be specified")
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        if args.all:
            # Run both exchanges in separate threads
            logger.info("Starting trading on all configured exchanges")
            
            threads = []
            
            # Start Kraken thread
            if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"):
                kraken_thread = threading.Thread(
                    target=run_kraken,
                    name="KrakenTradingLoop",
                    daemon=True
                )
                threads.append(kraken_thread)
                kraken_thread.start()
                logger.info("Started Kraken trading thread")
            
            # Start Binance.US thread
            if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
                binance_thread = threading.Thread(
                    target=run_binanceus,
                    name="BinanceUSTradingLoop",
                    daemon=True
                )
                threads.append(binance_thread)
                binance_thread.start()
                logger.info("Started Binance.US trading thread")
            
            if not threads:
                logger.error("No exchanges configured. Please check your API keys.")
                sys.exit(1)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                # Threads are daemon threads, so they'll exit when main does
                
        else:  # Single exchange mode
            logger.info(f"Starting trading on {args.exchange}")
            
            if args.exchange == "kraken":
                run_kraken()
            elif args.exchange == "binanceus":
                run_binanceus()
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
