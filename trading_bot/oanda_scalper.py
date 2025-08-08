import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from oanda_engine import OandaTradingEngine
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('oanda_scalper.log')
    ]
)
logger = logging.getLogger('OandaScalper')

class OandaScalper:
    """Scalping bot for OANDA with 10% position sizing."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv('.env.oanda')
        
        # Initialize trading engine
        self.engine = OandaTradingEngine()
        self.trading_pair = os.getenv('TRADING_PAIR', 'EUR_USD')
        
        # Strategy parameters
        self.ema_fast = 9
        self.ema_medium = 21
        self.ema_slow = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # FastAPI setup
        self.app = FastAPI(title="OANDA Scalping Bot")
        self.setup_routes()
        
        # State
        self.last_signal = None
        self.last_update = None
        self.account_info = {}
        
        logger.info(f"OANDA Scalper initialized for {self.trading_pair}")
    
    def setup_routes(self):
        """Set up API endpoints."""
        @self.app.get("/")
        async def root():
            return {"status": "running", 
                   "pair": self.trading_pair,
                   "last_signal": self.last_signal,
                   "last_update": self.last_update}
        
        @self.app.get("/status")
        async def status():
            return await self.get_status()
        
        @self.app.post("/webhook")
        async def webhook(request: Request):
            data = await request.json()
            logger.info(f"Received webhook: {data}")
            return {"status": "received"}
        
        @self.app.post("/close_all")
        async def close_all():
            success = await self.engine.close_all_positions()
            return {"status": "success" if success else "error",
                   "message": "All positions closed" if success else "Error closing positions"}
    
    async def get_status(self) -> Dict:
        """Get current bot status."""
        balance = await self.engine.get_account_balance()
        price = await self.engine.get_current_price()
        
        return {
            "status": "online",
            "trading_pair": self.trading_pair,
            "account_balance": balance,
            "current_price": price,
            "last_signal": self.last_signal,
            "last_update": self.last_update
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # EMAs
            df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
            df['ema_medium'] = df['close'].ewm(span=self.ema_medium, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # ATR for volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    async def analyze_market(self) -> Optional[Dict]:
        """Analyze market conditions and generate trading signals."""
        try:
            # Get historical data (last 200 candles)
            df = await self.engine.get_historical_data(count=200, granularity="M5")
            if df is None or df.empty:
                logger.warning("No historical data available")
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                return None
            
            # Get the latest values
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize signal
            signal = {
                'timestamp': datetime.utcnow().isoformat(),
                'pair': self.trading_pair,
                'price': current['close'],
                'action': 'hold',
                'confidence': 0.0,
                'indicators': {
                    'ema_fast': current['ema_fast'],
                    'ema_medium': current['ema_medium'],
                    'ema_slow': current['ema_slow'],
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'atr': current['atr']
                }
            }
            
            # Check for buy signals
            buy_conditions = [
                current['ema_fast'] > current['ema_medium'] > current['ema_slow'],
                current['macd'] > current['macd_signal'],
                current['rsi'] < self.rsi_overbought,
                current['close'] > current['ema_medium']
            ]
            
            # Check for sell signals
            sell_conditions = [
                current['ema_fast'] < current['ema_medium'] < current['ema_slow'],
                current['macd'] < current['macd_signal'],
                current['rsi'] > self.rsi_oversold,
                current['close'] < current['ema_medium']
            ]
            
            # Calculate confidence (percentage of conditions met)
            buy_confidence = sum(buy_conditions) / len(buy_conditions)
            sell_confidence = sum(sell_conditions) / len(sell_conditions)
            
            # Determine signal
            min_confidence = 0.7  # 70% of conditions must be met
            
            if buy_confidence >= min_confidence and buy_confidence > sell_confidence:
                signal['action'] = 'buy'
                signal['confidence'] = buy_confidence
            elif sell_confidence >= min_confidence and sell_confidence > buy_confidence:
                signal['action'] = 'sell'
                signal['confidence'] = sell_confidence
            
            self.last_signal = signal
            self.last_update = datetime.utcnow().isoformat()
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}", exc_info=True)
            return None
    
    async def execute_trading_cycle(self):
        """Execute one trading cycle (analysis + execution)."""
        try:
            # Check if market is open
            if not self.engine.is_market_open():
                logger.info("Market is closed, skipping trading cycle")
                return
            
            # Get account balance for position sizing
            balance = await self.engine.get_account_balance()
            if balance <= 0:
                logger.error(f"Invalid account balance: {balance}")
                return
            
            # Analyze market
            signal = await self.analyze_market()
            if signal is None or signal['action'] == 'hold':
                logger.debug("No trading signal generated")
                return
            
            # Execute trade
            logger.info(f"Executing {signal['action']} signal with {signal['confidence']*100:.1f}% confidence")
            
            # Place the trade with 10% position sizing (handled by OandaTradingEngine)
            trade_result = await self.engine.place_trade(signal['action'], signal['price'])
            
            if trade_result:
                logger.info(f"Trade executed: {trade_result}")
            else:
                logger.warning("Failed to execute trade")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
    
    async def run_strategy(self, interval: int = 300):
        """Run the trading strategy in a loop."""
        logger.info(f"Starting trading strategy with {interval}s interval")
        
        while True:
            try:
                await self.execute_trading_cycle()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Trading strategy stopped")
                break
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying

# Run the bot
async def main():
    # Initialize the scalper
    scalper = OandaScalper()
    
    # Start the trading strategy in the background
    strategy_task = asyncio.create_task(scalper.run_strategy())
    
    # Start the FastAPI server
    config = uvicorn.Config(
        app=scalper.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        strategy_task.cancel()
        await strategy_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
