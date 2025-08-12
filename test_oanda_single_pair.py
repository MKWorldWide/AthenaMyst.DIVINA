"""
Enhanced test script for Oanda trading bot with a single currency pair.
This script focuses on detailed logging and error handling for debugging.
"""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.oanda_engine import OandaTradingEngine

# Configure logging with detailed formatting
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler with color formatting
class ColorConsoleHandler(logging.StreamHandler):
    """Custom console handler with color formatting."""
    COLOR_CODES = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelname in self.COLOR_CODES:
                msg = f"{self.COLOR_CODES[record.levelname]}{msg}{self.COLOR_CODES['RESET']}"
            print(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add color console handler
console_handler = ColorConsoleHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# Add file handler with more detailed logging
file_handler = logging.FileHandler('oanda_single_pair_test.log', mode='w')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger('OandaTest')

# Reduce log level for oandapyV20 to reduce noise
logging.getLogger('oandapyV20').setLevel(logging.WARNING)

class OandaTester:
    """Test harness for Oanda trading bot functionality."""
    
    def __init__(self, test_pairs: List[str] = None):
        """Initialize the test harness."""
        self.test_pairs = test_pairs or ['EUR_USD']  # Default to EUR/USD for single-pair testing
        self.engine = None
        self.test_start_time = datetime.now()
        self.test_timeout = 300  # 5 minutes default timeout
        self.iteration_count = 0
        self.max_iterations = 5  # Number of test iterations to run
        
        logger.info("=" * 80)
        logger.info("OANDA SINGLE-PAIR TRADING TEST")
        logger.info("=" * 80)
        logger.info(f"Starting test at {self.test_start_time.isoformat()}")
        logger.info(f"Test pairs: {', '.join(self.test_pairs)}")
        logger.info(f"Test timeout: {self.test_timeout} seconds")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info("-" * 80)
    
    async def initialize_engine(self) -> bool:
        """Initialize the Oanda trading engine."""
        try:
            logger.info("Initializing OandaTradingEngine...")
            
            # Initialize with the first test pair only for single-pair testing
            self.engine = OandaTradingEngine(
                config_file='.env.oanda',
                dry_run=True,  # Always run in dry-run mode for testing
                trading_pairs=[self.test_pairs[0]]  # Single pair for this test
            )
            
            # Verify engine initialization
            if not hasattr(self.engine, 'client') or self.engine.client is None:
                raise RuntimeError("OANDA client not properly initialized")
                
            # Log engine configuration
            logger.info("OandaTradingEngine initialized successfully")
            logger.info(f"Account ID: {getattr(self.engine, 'account_id', 'Not set')}")
            logger.info(f"Account Type: {getattr(self.engine, 'account_type', 'Not set')}")
            logger.info(f"Dry Run Mode: {getattr(self.engine, 'dry_run', 'Not set')}")
            logger.info(f"Trading Pairs: {getattr(self.engine, 'trading_pairs', 'Not set')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OandaTradingEngine: {e}", exc_info=True)
            logger.error("Please check your .env.oanda file and OANDA credentials")
            return False
    
    async def test_price_fetching(self) -> bool:
        """Test fetching prices for the test pair."""
        logger.info("\n=== Testing Price Fetching ===")
        success = True
        
        for pair in self.test_pairs[:1]:  # Only test first pair
            try:
                start_time = time.time()
                price = await self.engine.get_current_price(pair)
                elapsed = (time.time() - start_time) * 1000  # ms
                
                if price is not None and price > 0:
                    logger.info(f"✓ {pair}: {price:.5f} (took {elapsed:.2f}ms)")
                else:
                    logger.error(f"✗ {pair}: Failed to get valid price (got {price})")
                    success = False
                    
            except Exception as e:
                logger.error(f"✗ {pair}: Error fetching price: {e}", exc_info=True)
                success = False
                
        return success
    
    async def test_signal_generation(self) -> bool:
        """Test signal generation for the test pair."""
        logger.info("\n=== Testing Signal Generation ===")
        success = True
        
        for pair in self.test_pairs[:1]:  # Only test first pair
            try:
                start_time = time.time()
                signal = await self.engine.generate_signal(pair)
                elapsed = (time.time() - start_time) * 1000  # ms
                
                if signal in ['BUY', 'SELL', 'HOLD']:
                    indicators = self.engine.pair_states[pair].get('indicators', {})
                    logger.info(f"✓ {pair}: {signal} | "
                              f"Price: {indicators.get('price', 'N/A'):.5f} | "
                              f"SMA20: {indicators.get('sma20', 'N/A'):.5f} | "
                              f"SMA50: {indicators.get('sma50', 'N/A'):.5f} | "
                              f"RSI: {indicators.get('rsi', 'N/A'):.2f} | "
                              f"(took {elapsed:.2f}ms)")
                else:
                    logger.error(f"✗ {pair}: Invalid signal generated: {signal}")
                    success = False
                    
            except Exception as e:
                logger.error(f"✗ {pair}: Error generating signal: {e}", exc_info=True)
                success = False
                
        return success
    
    async def test_trading_cycle(self) -> bool:
        """Test a complete trading cycle (price fetch + signal generation)."""
        logger.info("\n=== Testing Trading Cycle ===")
        self.iteration_count += 1
        
        # Log iteration header
        logger.info(f"\n{'='*40} Iteration {self.iteration_count} {'='*40}")
        
        # Test price fetching
        price_success = await self.test_price_fetching()
        if not price_success:
            logger.error("Price fetching test failed, aborting trading cycle")
            return False
        
        # Test signal generation
        signal_success = await self.test_signal_generation()
        if not signal_success:
            logger.error("Signal generation test failed")
            return False
            
        return True
    
    async def run_tests(self):
        """Run the test sequence."""
        try:
            # Initialize the trading engine
            if not await self.initialize_engine():
                return False
            
            # Run test iterations
            test_success = True
            for i in range(self.max_iterations):
                try:
                    iteration_success = await asyncio.wait_for(
                        self.test_trading_cycle(),
                        timeout=30  # 30 seconds per iteration
                    )
                    
                    if not iteration_success:
                        test_success = False
                        logger.warning(f"Iteration {i+1} failed")
                    
                    # Small delay between iterations
                    if i < self.max_iterations - 1:  # No need to sleep after last iteration
                        await asyncio.sleep(5)
                        
                except asyncio.TimeoutError:
                    logger.error(f"Iteration {i+1} timed out after 30 seconds")
                    test_success = False
                    break
                except Exception as e:
                    logger.error(f"Error in iteration {i+1}: {e}", exc_info=True)
                    test_success = False
                    break
            
            # Log test results
            test_duration = (datetime.now() - self.test_start_time).total_seconds()
            logger.info("\n" + "="*80)
            logger.info("TEST SUMMARY")
            logger.info("="*80)
            logger.info(f"Test completed in {test_duration:.2f} seconds")
            logger.info(f"Iterations completed: {self.iteration_count}/{self.max_iterations}")
            logger.info(f"Final status: {'PASSED' if test_success else 'FAILED'}")
            
            return test_success
            
        except Exception as e:
            logger.error(f"Test failed with unexpected error: {e}", exc_info=True)
            return False

async def main():
    """Main test function."""
    # Start with a single pair for initial testing
    test_pairs = ['EUR_USD']
    
    # Run the tests
    tester = OandaTester(test_pairs)
    success = await tester.run_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
