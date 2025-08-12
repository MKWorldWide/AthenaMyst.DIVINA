import asyncio
import pandas as pd
from trading_bot.oanda_engine import OandaTradingEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_historical_data():
    """Test historical data retrieval for OANDA."""
    try:
        # Initialize the trading engine
        engine = OandaTradingEngine(dry_run=True)
        
        # Test with default pair (EUR_USD)
        logger.info("Testing historical data for default pair...")
        df = await engine.get_historical_data(count=5, granularity="M5")
        if df is not None and not df.empty:
            logger.info(f"Successfully retrieved {len(df)} candles for default pair:")
            print(df.tail())
        else:
            logger.error("Failed to retrieve historical data for default pair")
        
        # Test with specific pair
        test_pair = "USD_JPY"
        logger.info(f"\nTesting historical data for {test_pair}...")
        df = await engine.get_historical_data(pair=test_pair, count=5, granularity="M5")
        if df is not None and not df.empty:
            logger.info(f"Successfully retrieved {len(df)} candles for {test_pair}:")
            print(df.tail())
        else:
            logger.error(f"Failed to retrieve historical data for {test_pair}")
            
    except Exception as e:
        logger.error(f"Error in test_historical_data: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_historical_data())
