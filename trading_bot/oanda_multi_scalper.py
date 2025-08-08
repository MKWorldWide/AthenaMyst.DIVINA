import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from oanda_engine import OandaTradingEngine
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('oanda_multi_scalper.log')
    ]
)
logger = logging.getLogger('OandaMultiScalper')

class OandaMultiScalper:
    """Multi-pair scalping bot for OANDA with 10% position sizing."""
    
    def __init__(self, config_file: str = '.env.oanda.multi', discord_webhook_url: str = None):
        # Load configuration
        load_dotenv(config_file)
        
        # Initialize Discord webhook
        self.discord_webhook_url = discord_webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        if self.discord_webhook_url:
            logger.info("Discord webhook URL configured")
        
        # Initialize trading engine for each pair
        self.trading_pairs = os.getenv('TRADING_PAIRS', 'EUR_USD,GBP_JPY').split(',')
        self.engines = {}
        self.active_trades = {pair: None for pair in self.trading_pairs}
        
        # Initialize FastAPI app
        self.app = FastAPI(title="OANDA Multi-Pair Scalper")
        self.setup_routes()
        
        # Load configuration for each pair
        for pair in self.trading_pairs:
            # Create a copy of the environment variables for this pair
            pair_config = {
                'OANDA_ACCOUNT_ID': os.getenv('OANDA_ACCOUNT_ID'),
                'OANDA_API_KEY': os.getenv('OANDA_API_KEY'),
                'OANDA_ACCOUNT_TYPE': os.getenv('OANDA_ACCOUNT_TYPE', 'live'),
                'TRADING_PAIR': pair,
                'ACCOUNT_CURRENCY': os.getenv('ACCOUNT_CURRENCY', 'USD'),
                'RISK_PERCENT': str(float(os.getenv('RISK_PERCENT', '10.0')) / len(self.trading_pairs)),
                'STOP_LOSS_PIPS': os.getenv('STOP_LOSS_PIPS', '20'),
                'TAKE_PROFIT_PIPS': os.getenv('TAKE_PROFIT_PIPS', '40'),
                'DISCORD_WEBHOOK_URL': self.discord_webhook_url
            }
            
            # Save to a temporary config file for this pair
            config_filename = f'.env.oanda.{pair}'
            with open(config_filename, 'w') as f:
                for key, value in pair_config.items():
                    f.write(f"{key}={value}\n")
            
            # Initialize engine
            self.engines[pair] = OandaTradingEngine(
                config_file=config_filename,
                discord_webhook_url=self.discord_webhook_url
            )
            logger.info(f"Initialized engine for {pair}")
        
        # Strategy parameters
        self.ema_fast = 9
        self.ema_medium = 21
        self.ema_slow = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Trading limits
        self.max_open_trades = int(os.getenv('MAX_OPEN_TRADES', '2'))
        self.max_trades_per_day = int(os.getenv('MAX_TRADES_PER_DAY', '5'))
        self.today_trades = {pair: 0 for pair in self.trading_pairs}
        self.last_trade_day = datetime.utcnow().date()
        
        # FastAPI setup
        self.app = FastAPI(title="OANDA Multi-Pair Scalping Bot")
        self.setup_routes()
        
        logger.info(f"OANDA Multi-Pair Scalper initialized for pairs: {', '.join(self.trading_pairs)}")
    
    def setup_routes(self):
        """Set up API endpoints."""
        @self.app.get("/")
        async def root():
            return {"status": "running", 
                   "pairs": self.trading_pairs,
                   "active_trades": self.active_trades}
        
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
            success = await self.close_all_positions()
            return {"status": "success" if success else "error",
                   "message": "All positions closed" if success else "Error closing positions"}
    
    async def get_status(self) -> Dict:
        """Get current bot status."""
        status = {
            "status": "online",
            "pairs": self.trading_pairs,
            "active_trades": {},
            "account_info": {},
            "today_trades": self.today_trades
        }
        
        # Get status for each pair
        for pair, engine in self.engines.items():
            try:
                balance = await engine.get_account_balance()
                price = await engine.get_current_price()
                status["account_info"][pair] = {
                    "balance": balance,
                    "current_price": price,
                    "active_trade": self.active_trades[pair] is not None
                }
            except Exception as e:
                logger.error(f"Error getting status for {pair}: {e}")
                status["account_info"][pair] = {"error": str(e)}
        
        return status
    
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Analyze market conditions for a specific pair."""
        try:
            engine = self.engines[pair]
            
            # Skip if we already have an active trade for this pair
            if self.active_trades[pair] is not None:
                return None
            
            # Get historical data
            df = await engine.get_historical_data(count=200, granularity="M5")
            if df is None or df.empty:
                logger.warning(f"No historical data available for {pair}")
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                return None
            
            # Get the latest values
            current = df.iloc[-1]
            
            # Initialize signal
            signal = {
                'pair': pair,
                'timestamp': datetime.utcnow().isoformat(),
                'price': current['close'],
                'action': 'hold',
                'confidence': 0.0,
                'indicators': {
                    'ema_fast': current['ema_fast'],
                    'ema_medium': current['ema_medium'],
                    'ema_slow': current['ema_slow'],
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal']
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
            
            # Calculate confidence
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
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
            return None
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on the signal."""
        try:
            pair = signal['pair']
            action = signal['action']
            price = signal['price']
            confidence = signal['confidence']
            
            # Skip if no valid signal
            if action not in ['buy', 'sell']:
                return False
            
            # Check if we already have an active trade for this pair
            if self.active_trades[pair] is not None:
                logger.info(f"Skipping {pair} - already have an active trade")
                return False
            
            # Check if we've hit the daily trade limit for this pair
            if self.today_trades[pair] >= self.max_trades_per_day:
                logger.info(f"Skipping {pair} - daily trade limit reached")
                return False
            
            # Count the number of currently active trades
            active_trade_count = sum(1 for t in self.active_trades.values() if t is not None)
            if active_trade_count >= self.max_open_trades:
                logger.info("Skipping trade - maximum open trades reached")
                return False
            
            logger.info(f"Executing {action.upper()} signal for {pair} with {confidence*100:.1f}% confidence")
            
            # Place the trade
            engine = self.engines[pair]
            trade_result = await engine.place_trade(action, price)
            
            if trade_result:
                self.active_trades[pair] = {
                    'id': trade_result.get('orderFillTransaction', {}).get('id', 'unknown'),
                    'pair': pair,
                    'direction': action,
                    'entry_price': price,
                    'timestamp': datetime.utcnow().isoformat(),
                    'stop_loss': trade_result.get('orderFillTransaction', {}).get('stopLossOnFill', {}).get('price'),
                    'take_profit': trade_result.get('orderFillTransaction', {}).get('takeProfitOnFill', {}).get('price')
                }
                self.today_trades[pair] += 1
                logger.info(f"Trade executed for {pair}: {trade_result}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False
    
    async def check_open_trades(self):
        """Check and update status of open trades."""
        try:
            for pair, trade in list(self.active_trades.items()):
                if trade is None:
                    continue
                
                try:
                    # Check if the trade is still open
                    engine = self.engines[pair]
                    open_trades = await engine.get_open_trades()
                    
                    # Check if our trade is still open
                    trade_still_open = False
                    for open_trade in open_trades:
                        if str(open_trade.get('id')) == str(trade.get('id')):
                            trade_still_open = True
                            break
                    
                    # If the trade is no longer open, update our records
                    if not trade_still_open:
                        logger.info(f"Trade closed for {pair}")
                        self.active_trades[pair] = None
                        
                except Exception as e:
                    logger.error(f"Error checking trade status for {pair}: {e}")
                    
        except Exception as e:
            logger.error(f"Error checking open trades: {e}", exc_info=True)
    
    async def close_all_positions(self) -> bool:
        """Close all open positions."""
        success = True
        for pair, engine in self.engines.items():
            try:
                await engine.close_all_positions()
                self.active_trades[pair] = None
                logger.info(f"Closed all positions for {pair}")
            except Exception as e:
                logger.error(f"Error closing positions for {pair}: {e}")
                success = False
        return success
    
    async def reset_daily_counts(self):
        """Reset daily trade counts at the start of a new day."""
        now = datetime.utcnow()
        if now.date() > self.last_trade_day:
            logger.info("Resetting daily trade counts")
            self.today_trades = {pair: 0 for pair in self.trading_pairs}
            self.last_trade_day = now.date()
    
    async def trading_cycle(self):
        """Execute one trading cycle (analysis + execution)."""
        try:
            # Reset daily counts if needed
            await self.reset_daily_counts()
            
            # Check and update open trades
            await self.check_open_trades()
            
            # Analyze each pair and execute trades
            for pair in self.trading_pairs:
                try:
                    signal = await self.analyze_pair(pair)
                    if signal and signal['action'] in ['buy', 'sell']:
                        await self.execute_trade(signal)
                    await asyncio.sleep(1)  # Small delay between pairs
                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
    
    async def run_strategy(self, interval: int = 300):
        """Run the trading strategy in a loop."""
        logger.info(f"Starting trading strategy with {interval}s interval")
        
        while True:
            try:
                await self.trading_cycle()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Trading strategy stopped")
                break
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying

def run_bot():
    # Your Discord webhook URL
    DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1403240581882318933/5CK8zu_m7jgPega67C9OGlA5FnQuVKhf2kEyr2x-acRoONEG290YT2U8d4pFbi5VI3Yl"
    
    # Create and run the scalper
    scalper = OandaMultiScalper(
        config_file='.env.oanda.multi',
        discord_webhook_url=DISCORD_WEBHOOK_URL
    )
    
    # Run the FastAPI server in a separate thread
    import threading
    
    def run_server():
        uvicorn.run(
            app=scalper.app,
            host="0.0.0.0",
            port=8003,
            log_level="info"
        )
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Start the trading bot
    asyncio.run(scalper.run_strategy())

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
