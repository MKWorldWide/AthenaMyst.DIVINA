import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_bot.oanda_engine import OandaTradingEngine
import logging
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('signal_analysis.log')
    ]
)
logger = logging.getLogger('SignalAnalyzer')

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        """
        Rate limiter to control API call frequency.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max(max_calls, 1)  # Ensure at least 1 call is allowed
        self.period = max(period, 1.0)  # Ensure minimum period of 1 second
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait until we can make another API call."""
        async with self.lock:
            now = time.time()
            
            # Remove calls older than the period
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # If we've reached the rate limit, wait until we can make another call
            if len(self.calls) >= self.max_calls:
                sleep_time = max(0.0, self.period - (now - self.calls[0]) + 0.1)  # Add small buffer
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                    now = time.time()  # Update now after sleeping
            
            self.calls.append(now)
            
            # Log rate limit status
            current_rate = len(self.calls) / self.period
            logger.debug(f"Rate: {current_rate:.2f} calls/sec (limit: {self.max_calls/self.period:.2f})")

class SignalAnalyzer:
    def __init__(self, engine: OandaTradingEngine, max_requests_per_minute: int = 60):
        self.engine = engine
        self.signals = []
        self.rate_limiter = RateLimiter(
            max_calls=max_requests_per_minute,
            period=60  # 1 minute
        )
        
    async def get_historical_data_with_retry(self, pair: str, count: int, granularity: str, retries: int = 3) -> pd.DataFrame:
        """Get historical data with retry logic and rate limiting."""
        last_error = None
        
        for attempt in range(retries):
            try:
                await self.rate_limiter.acquire()
                logger.debug(f"Fetching {count} {granularity} candles for {pair} (attempt {attempt + 1}/{retries})")
                df = await self.engine.get_historical_data(
                    pair=pair,
                    count=count,
                    granularity=granularity
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_error = e
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed for {pair}: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to fetch data for {pair} after {retries} attempts")
        if last_error:
            logger.error(f"Last error: {str(last_error)}")
        return pd.DataFrame()
    
    async def analyze_pairs(self, pairs: list, days: int = 1, timeframe: str = "M5") -> dict:
        """Analyze signals for multiple pairs over a number of days."""
        results = {}
        start_time = time.time()
        total_pairs = len(pairs)
        
        for idx, pair in enumerate(pairs, 1):
            try:
                elapsed = (time.time() - start_time) / 60  # in minutes
                remaining_pairs = total_pairs - idx + 1
                avg_time_per_pair = elapsed / (idx or 1)
                eta = avg_time_per_pair * remaining_pairs
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Analyzing {pair} ({idx}/{total_pairs}) - {elapsed:.1f} min elapsed, ~{eta:.1f} min remaining")
                logger.info(f"{'='*70}")
                
                # Log rate limit status
                current_rate = len(self.rate_limiter.calls) / self.rate_limiter.period if hasattr(self.rate_limiter, 'calls') else 0
                logger.info(f"API Rate: {current_rate:.2f} calls/sec (limit: {self.rate_limiter.max_calls/self.rate_limiter.period:.2f})")
                
                # Calculate number of candles needed (5-min candles, 288 per day)
                candles_per_day = 288  # 24h * 60min / 5min
                count = min(1000, days * candles_per_day)
                
                # Get historical data with rate limiting and retries
                df = await self.get_historical_data_with_retry(
                    pair=pair,
                    count=count,
                    granularity=timeframe
                )
                
                if df.empty:
                    logger.warning(f"No data available for {pair}")
                    continue
                
                # Generate signals for each time period
                signals = []
                for i in range(14, len(df)):  # Start from 14 to have enough data for RSI
                    try:
                        # Get the current data window
                        window = df.iloc[:i+1].copy()
                        
                        # Generate signal with rate limiting
                        await self.rate_limiter.acquire()
                        signal = await self.engine.generate_signal(pair=pair)
                        
                        # Record the signal with price data
                        if signal and signal != 'HOLD':
                            signals.append({
                                'timestamp': window.index[-1],
                                'pair': pair,
                                'signal': signal,
                                'price': window['close'].iloc[-1],
                                'rsi': window['rsi'].iloc[-1] if 'rsi' in window.columns else None
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing {pair} at index {i}: {e}")
                        continue
                
                # Save results
                if signals:
                    results[pair] = pd.DataFrame(signals)
                    logger.info(f"Generated {len(signals)} signals for {pair}")
                    logger.info(f"Signal distribution:\n{results[pair]['signal'].value_counts()}")
                else:
                    logger.warning(f"No signals generated for {pair}")
                
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
                continue
                
        return results
    
    def plot_signals(self, results: dict):
        """Plot price and signals for each pair."""
        if not results:
            logger.warning("No results to plot")
            return
            
        for pair, signals in results.items():
            try:
                if signals.empty:
                    continue
                    
                plt.figure(figsize=(12, 8))
                
                # Plot price
                plt.subplot(2, 1, 1)
                plt.plot(signals['timestamp'], signals['price'], label='Price', color='blue')
                
                # Plot buy signals
                buy_signals = signals[signals['signal'] == 'BUY']
                if not buy_signals.empty:
                    plt.scatter(buy_signals['timestamp'], buy_signals['price'], 
                               color='green', marker='^', label='Buy', s=100)
                
                # Plot sell signals
                sell_signals = signals[signals['signal'] == 'SELL']
                if not sell_signals.empty:
                    plt.scatter(sell_signals['timestamp'], sell_signals['price'], 
                               color='red', marker='v', label='Sell', s=100)
                
                plt.title(f'Price and Signals - {pair}')
                plt.legend()
                plt.grid(True)
                
                # Plot RSI if available
                if 'rsi' in signals.columns:
                    plt.subplot(2, 1, 2)
                    plt.plot(signals['timestamp'], signals['rsi'], label='RSI', color='purple')
                    plt.axhline(70, color='red', linestyle='--', alpha=0.3)
                    plt.axhline(30, color='green', linestyle='--', alpha=0.3)
                    plt.title('RSI')
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'signals_{pair.replace("/", "_")}.png')
                logger.info(f"Saved signal plot for {pair} to signals_{pair.replace('/', '_')}.png")
                
            except Exception as e:
                logger.error(f"Error plotting {pair}: {e}")
                continue

async def main():
    try:
        logger.info("Starting signal analysis...")
        
        # Initialize the trading engine with test pairs
        test_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
        
        logger.info(f"Initializing OandaTradingEngine with pairs: {test_pairs}")
        engine = OandaTradingEngine(
            dry_run=True,
            trading_pairs=test_pairs
        )
        
        # Initialize signal analyzer with rate limiting (30 requests per minute)
        analyzer = SignalAnalyzer(engine, max_requests_per_minute=30)
        
        # Analyze signals for the last 3 days with 5-minute candles
        logger.info("Starting signal analysis...")
        results = await analyzer.analyze_pairs(test_pairs, days=3, timeframe="M5")
        
        # Plot results
        if results:
            analyzer.plot_signals(results)
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'engine' in locals():
            logger.info("Closing OandaTradingEngine...")
            await engine.close()
            
        logger.info("Signal analysis completed")

if __name__ == "__main__":
    asyncio.run(main())
