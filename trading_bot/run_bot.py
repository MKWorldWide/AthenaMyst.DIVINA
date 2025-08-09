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

from trading_loop import run_kraken, run_binanceus

def setup_logging(exchange: str = None) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        exchange: Exchange name (e.g., 'kraken', 'binanceus') or None for main logger
        
    Returns:
        Configured logger instance
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a logger specific to the exchange or use the root logger
    if exchange:
        logger = logging.getLogger(f"trading_bot.{exchange}")
        log_file = os.path.join(logs_dir, f"{exchange}.log")
    else:
        logger = logging.getLogger()
        log_file = os.path.join(logs_dir, "trading_bot.log")
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set log level
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return logger

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
    
    # Setup main logger
    logger = setup_logging()
    logger.info("Trading bot starting...")
    
    try:
        if args.all:
            # Run both exchanges in separate threads
            logger.info("Starting trading on all configured exchanges")
            
            threads = []
            
            # Start Kraken thread
            if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"):
                kraken_logger = setup_logging("kraken")
                kraken_thread = threading.Thread(
                    target=run_kraken,
                    name="KrakenTradingLoop",
                    daemon=True,
                    kwargs={"logger": kraken_logger}
                )
                threads.append(kraken_thread)
                kraken_thread.start()
                logger.info("Started Kraken trading thread")
            
            # Start Binance.US thread
            if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
                binance_logger = setup_logging("binanceus")
                binance_thread = threading.Thread(
                    target=run_binanceus,
                    name="BinanceUSTradingLoop",
                    daemon=True,
                    kwargs={"logger": binance_logger}
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
            exchange_logger = setup_logging(args.exchange)
            logger.info(f"Starting trading on {args.exchange}")
            
            if args.exchange == "kraken":
                run_kraken(logger=exchange_logger)
            elif args.exchange == "binanceus":
                run_binanceus(logger=exchange_logger)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
