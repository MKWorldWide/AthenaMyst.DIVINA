import asyncio
import logging
from trading_bot.oanda_engine import OandaTradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_multi_pair_signals():
    """Test signal generation for multiple currency pairs."""
    try:
        # Initialize the trading engine with test pairs
        test_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
        engine = OandaTradingEngine(
            dry_run=True,
            trading_pairs=test_pairs
        )
        
        logger.info(f"Testing signal generation for pairs: {test_pairs}")
        
        # Test signal generation for each pair
        for pair in test_pairs:
            try:
                logger.info(f"\nGenerating signal for {pair}...")
                signal = await engine.generate_signal(pair=pair)
                
                if signal:
                    logger.info(f"Signal for {pair}: {signal}")
                    
                    # Get historical data for context
                    df = await engine.get_historical_data(pair=pair, count=10, granularity="M5")
                    if df is not None and not df.empty:
                        logger.info(f"Latest price for {pair}: {df['close'].iloc[-1]}")
                else:
                    logger.warning(f"No signal generated for {pair}")
                    
            except Exception as e:
                logger.error(f"Error testing {pair}: {e}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Error in test_multi_pair_signals: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_multi_pair_signals())
