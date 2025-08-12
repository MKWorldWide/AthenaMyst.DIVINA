import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_bot.oanda_engine import OandaTradingEngine
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalAnalyzer:
    def __init__(self, engine: OandaTradingEngine):
        self.engine = engine
        self.signals = []
        
    async def analyze_pairs(self, pairs: list, days: int = 1):
        """Analyze signals for multiple pairs over a number of days."""
        results = {}
        
        for pair in pairs:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Analyzing {pair}")
                logger.info(f"{'='*50}")
                
                # Get historical data
                df = await self.engine.get_historical_data(
                    pair=pair,
                    count=min(1000, days * 288),  # ~288 5-minute candles per day
                    granularity="M5"
                )
                
                if df is None or df.empty:
                    logger.warning(f"No data for {pair}")
                    continue
                
                # Generate signals for each time period
                signals = []
                for i in range(14, len(df)):  # Start from 14 to have enough data for RSI
                    try:
                        # Get the current data window
                        window = df.iloc[:i+1].copy()
                        
                        # Generate signal
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
                plt.show()
                
            except Exception as e:
                logger.error(f"Error plotting {pair}: {e}")
                continue

async def main():
    try:
        # Initialize the trading engine with test pairs
        test_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
        engine = OandaTradingEngine(
            dry_run=True,
            trading_pairs=test_pairs
        )
        
        # Initialize signal analyzer
        analyzer = SignalAnalyzer(engine)
        
        # Analyze signals for the last 3 days
        logger.info("Starting signal analysis...")
        results = await analyzer.analyze_pairs(test_pairs, days=3)
        
        # Plot results
        if results:
            analyzer.plot_signals(results)
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'engine' in locals():
            await engine.close()

if __name__ == "__main__":
    asyncio.run(main())
