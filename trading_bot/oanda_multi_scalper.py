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
    
    def setup_logging(self) -> None:
        """Initialize class-scoped logger and file/console handlers."""
        import logging
        import os
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger("OandaMultiScalper")
        self.logger.setLevel(logging.INFO)
        
        # Prevent adding multiple handlers if called more than once
        if not self.logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # File handler
            file_handler = logging.FileHandler('logs/oanda_multi_scalper.log')
            file_handler.setFormatter(formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("✅ Logging initialized")
    
    def _load_pairs(self) -> list[str]:
        """Load and validate trading pairs from environment or use defaults."""
        raw_pairs = os.getenv('TRADING_PAIRS', 'EUR_USD,GBP_JPY')
        pairs = [p.strip() for p in raw_pairs.split(',') if p.strip()]
        
        # Basic validation
        valid_pairs = []
        for pair in pairs:
            if '_' in pair and len(pair) >= 6:  # e.g., 'EUR_USD'
                valid_pairs.append(pair)
            else:
                self.logger.warning(f"Skipping invalid pair format: {pair}")
        
        if not valid_pairs:
            valid_pairs = ['EUR_USD', 'GBP_JPY']  # Fallback defaults
            self.logger.warning(f"No valid pairs found, using defaults: {valid_pairs}")
        
        return valid_pairs
    
    async def _init_engines(self) -> None:
        """Initialize trading engines for all valid pairs with detailed error handling."""
        self.engines = {}
        self.initialized_pairs = set()
        
        if not hasattr(self, 'pairs') or not self.pairs:
            self.logger.error("No trading pairs configured. Check TRADING_PAIRS in .env.oanda.multi")
            return
            
        self.logger.info(f"Starting engine initialization for {len(self.pairs)} pairs...")
        
        # Load main config first
        main_config = '.env.oanda.multi'
        if not os.path.exists(main_config):
            self.logger.error(f"Main config file not found: {os.path.abspath(main_config)}")
            return
            
        # Verify main config is readable
        try:
            load_dotenv(main_config)
            if not os.getenv('OANDA_API_KEY') or not os.getenv('OANDA_ACCOUNT_ID'):
                self.logger.error("Missing OANDA_API_KEY or OANDA_ACCOUNT_ID in main config")
                return
        except Exception as e:
            self.logger.error(f"Error loading main config: {str(e)}")
            return
            
        for pair in self.pairs:
            pair = pair.strip()
            if not pair:
                continue
                
            try:
                self.logger.info(f"Initializing engine for {pair}...")
                
                # Try pair-specific config first, fall back to main config
                config_filename = f'.env.oanda.{pair}'
                if not os.path.exists(config_filename):
                    self.logger.info(f"Using main config for {pair} (no pair-specific config found)")
                    config_filename = main_config
                
                self.logger.debug(f"Loading config from {os.path.abspath(config_filename)}")
                
                # Create engine instance
                engine = OandaTradingEngine(
                    config_file=config_filename,
                    discord_webhook_url=self.discord_webhook_url
                )
                
                # Test connection with timeout
                try:
                    self.logger.debug(f"Testing connection for {pair}...")
                    account_info = await asyncio.wait_for(
                        engine.get_account_info(),
                        timeout=10.0
                    )
                    
                    if not account_info:
                        raise Exception("Empty account info response")
                        
                    self.engines[pair] = engine
                    self.initialized_pairs.add(pair)
                    self.logger.info(f"✅ Successfully initialized engine for {pair}")
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"❌ Timeout initializing {pair}: Connection to OANDA API timed out")
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Connection test failed for {pair}: {str(e)}", exc_info=True)
                    continue
                
                # Small delay between initializations to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"❌ Unexpected error initializing {pair}: {str(e)}", exc_info=True)
                continue
        
        if not self.engines:
            self.logger.error("❌ No trading engines were successfully initialized")
            self.logger.error("Please check your configuration and network connection")
        else:
            self.logger.info(f"✅ Successfully initialized {len(self.engines)}/{len(self.pairs)} trading engines")
    
    async def initialize(self):
        """Initialize the multi-pair scalping bot asynchronously."""
        # Load environment variables
        load_dotenv('.env.oanda.multi')
        
        # Configure logging first
        self.setup_logging()
        self.logger.info("🚀 Starting Oanda Multi-Pair Scalper")
        
        # Load and validate pairs
        self.pairs = self._load_pairs()
        self.logger.info(f"Loaded {len(self.pairs)} trading pairs: {', '.join(self.pairs)}")
        
        # Initialize engines
        await self._init_engines()
        
        if not self.engines:
            self.logger.error("❌ No trading engines were successfully initialized")
            raise RuntimeError("No trading engines available")
            
        self.logger.info(f"✅ Successfully initialized {len(self.engines)}/{len(self.pairs)} trading pairs")
        
        # Initialize Discord webhook if configured
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if self.discord_webhook_url:
            logger.info("✅ Discord webhook URL configured")
        
        # Initialize trading engine for each pair with validation
        self.available_pairs = [pair.strip() for pair in os.getenv('TRADING_PAIRS', 'EUR_USD,GBP_JPY').split(',') if pair.strip()]
        self.engines = {}
        self.active_trades = {}
        self.pair_configs = {}  # Store configuration for each pair
        self.initialized_pairs = set()  # Track successfully initialized pairs
        
        # Validate and clean up pair list
        self.trading_pairs = []
        for pair in self.available_pairs:
            if '_' in pair and len(pair) >= 6:  # Basic validation (e.g., 'EUR_USD')
                self.trading_pairs.append(pair)
                self.active_trades[pair] = None
            else:
                logger.warning(f"Skipping invalid pair format: {pair}")
                
        if not self.trading_pairs:
            raise ValueError("No valid trading pairs configured. Please check TRADING_PAIRS in your .env file.")
        
        # Initialize FastAPI app
        self.app = FastAPI(title="OANDA Multi-Pair Scalper")
        self.setup_routes()
        
        return self
        
        # Load configuration for each pair with error handling
        for pair in self.trading_pairs:
            try:
                # Skip empty pairs
                if not pair or '_' not in pair:
                    logger.warning(f"Skipping invalid pair format: {pair}")
                    continue
                    
                # Create a copy of the environment variables for this pair
                risk_per_pair = float(os.getenv('RISK_PERCENT', '10.0')) / len(self.trading_pairs)
                pair_config = {
                    'OANDA_ACCOUNT_ID': os.getenv('OANDA_ACCOUNT_ID'),
                    'OANDA_API_KEY': os.getenv('OANDA_API_KEY'),
                    'OANDA_ACCOUNT_TYPE': os.getenv('OANDA_ACCOUNT_TYPE', 'live'),
                    'TRADING_PAIR': pair,
                    'ACCOUNT_CURRENCY': os.getenv('ACCOUNT_CURRENCY', 'USD'),
                    'RISK_PERCENT': str(risk_per_pair),
                    'STOP_LOSS_PIPS': os.getenv('STOP_LOSS_PIPS', '20'),
                    'TAKE_PROFIT_PIPS': os.getenv('TAKE_PROFIT_PIPS', '40'),
                    'DISCORD_WEBHOOK_URL': self.discord_webhook_url
                }
                self.pair_configs[pair] = pair_config
                
                # Save to a temporary config file for this pair
                config_filename = f'.env.oanda.{pair}'
                with open(config_filename, 'w') as f:
                    for key, value in pair_config.items():
                        if value is not None:  # Only write non-None values
                            f.write(f"{key}={value}\n")
                
                # Initialize engine with error handling
                try:
                    engine = OandaTradingEngine(
                        config_file=config_filename,
                        discord_webhook_url=self.discord_webhook_url
                    )
                    
                    # Test the connection for this pair
                    try:
                        # Test getting account information
                        account_info = await engine.get_account_info()
                        if not account_info:
                            raise Exception("Failed to get account info")
                            
                        # Test getting current price
                        price = await engine.get_current_price()
                        if not price:
                            raise Exception("Failed to get current price")
                            
                        # If we get here, initialization was successful
                        self.engines[pair] = engine
                        self.initialized_pairs.add(pair)
                        logger.info(f"✅ Successfully initialized {pair} with {risk_per_pair:.2f}% risk")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize {pair}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ Error initializing {pair}: {str(e)}")
                    continue
                    
                # Add a small delay between initializations to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Unexpected error initializing {pair}: {str(e)}")
                continue
        
        # Strategy parameters
        self.ema_fast = 9
        self.ema_medium = 21
        self.ema_slow = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Trading limits and pair management
        self.max_open_trades = int(os.getenv('MAX_OPEN_TRADES', '5'))  # Increased default for multiple pairs
        self.max_trades_per_day = int(os.getenv('MAX_TRADES_PER_DAY', '20'))  # Increased default for multiple pairs
        self.max_trades_per_pair = int(os.getenv('MAX_TRADES_PER_PAIR', '3'))  # New: Max trades per pair per day
        self.today_trades = {pair: 0 for pair in self.trading_pairs}
        self.last_trade_day = datetime.utcnow().date()
        
        # Trading hours for different sessions (UTC)
        self.trading_sessions = {
            'london': {'open': 7, 'close': 16},    # London session
            'new_york': {'open': 13, 'close': 22}, # New York session
            'tokyo': {'open': 0, 'close': 9},      # Tokyo session
            'sydney': {'open': 22, 'close': 7}     # Sydney session (wraps around midnight)
        }
        
        # FastAPI setup
        self.app = FastAPI(title="OANDA Multi-Pair Scalping Bot")
        self.setup_routes()
        
        self.logger.info(f"OANDA Multi-Pair Scalper initialized for pairs: {', '.join(self.pairs)}")
    
    def setup_routes(self):
        """Set up FastAPI routes."""
        from fastapi import HTTPException
        
        @self.app.get("/")
        async def root():
            """Root endpoint with basic bot info."""
            return {
                "service": "OANDA Multi-Pair Scalping Bot",
                "status": "running",
                "endpoints": [
                    "/health - Get service health status",
                    "/pairs - List all configured trading pairs",
                    "/pairs/reload - Reload pairs from environment"
                ]
            }
            
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring."""
            try:
                # Basic health checks
                if not hasattr(self, 'engines') or not self.engines:
                    raise HTTPException(status_code=503, detail="No trading engines available")
                        
                # Check each engine's connection
                status = {
                    "status": "healthy",
                    "initialized_pairs": list(self.initialized_pairs),
                    "total_engines": len(self.engines),
                    "active_trades": sum(1 for t in self.active_trades.values() if t is not None),
                    "timestamp": datetime.utcnow().isoformat()
                }
                    
                # Add per-engine status
                engine_status = {}
                for pair, engine in self.engines.items():
                    try:
                        # Quick check if engine is responsive
                        price = await engine.get_current_price()
                        engine_status[pair] = {
                            "status": "online",
                            "last_price": price,
                            "has_active_trade": self.active_trades.get(pair) is not None
                        }
                    except Exception as e:
                        engine_status[pair] = {
                            "status": "error",
                            "error": str(e)
                        }
                        status["status"] = "degraded"
                
                status["engines"] = engine_status
                return status
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
        
        @self.app.get("/pairs")
        async def list_pairs():
            """List all configured trading pairs and their status."""
            return {
                "configured_pairs": self.pairs,
                "initialized_pairs": list(self.initialized_pairs),
                "engines_available": list(self.engines.keys())
            }
            
        @self.app.post("/pairs/reload")
        async def reload_pairs():
            """Reload trading pairs from environment and reinitialize engines."""
            try:
                self.logger.info("Reloading trading pairs from environment...")
                
                # Store old state
                old_pairs = set(self.pairs)
                
                # Reload pairs
                self.pairs = self._load_pairs()
                
                # Reinitialize engines
                await self._init_engines()
                
                # Log changes
                new_pairs = set(self.pairs)
                added = new_pairs - old_pairs
                removed = old_pairs - new_pairs
                
                return {
                    "status": "success",
                    "message": "Trading pairs reloaded successfully",
                    "added_pairs": list(added),
                    "removed_pairs": list(removed),
                    "current_pairs": self.pairs,
                    "initialized_engines": list(self.engines.keys())
                }
                
            except Exception as e:
                self.logger.error(f"Failed to reload pairs: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to reload trading pairs: {str(e)}"
                )
        
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
        """Analyze a single currency pair."""
        try:
            # Skip if pair isn't initialized
            if pair not in self.initialized_pairs:
                logger.debug(f"Skipping analysis for uninitialized pair: {pair}")
                return None
                
            engine = self.engines.get(pair)
            if not engine:
                logger.error(f"No engine available for pair: {pair}")
                return None
            
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
    
    def is_market_open(self, pair: str) -> bool:
        """Check if the market is open for the given currency pair."""
        now = datetime.utcnow()
        hour = now.hour
        
        # Get the base currency (first 3 letters of the pair)
        base_currency = pair[:3].upper()
        
        # Determine which sessions are most relevant for this currency pair
        if base_currency in ['EUR', 'GBP', 'CHF']:
            # European pairs - focus on London session
            session = self.trading_sessions['london']
        elif base_currency in ['USD', 'CAD', 'MXN']:
            # American pairs - focus on New York session
            session = self.trading_sessions['new_york']
        elif base_currency in ['JPY', 'AUD', 'NZD']:
            # Asian pairs - focus on Tokyo/Sydney sessions
            session = self.trading_sessions['tokyo']
            # For AUD and NZD, also consider Sydney session
            if base_currency in ['AUD', 'NZD']:
                sydney = self.trading_sessions['sydney']
                if (hour >= sydney['open'] or hour < sydney['close']):
                    return True
        else:
            # Default to London/New York overlap (most liquid time)
            session = {'open': 12, 'close': 16}
        
        # Check if current time is within the session
        if session['open'] <= session['close']:
            return session['open'] <= hour < session['close']
        else:
            # Handle sessions that cross midnight
            return hour >= session['open'] or hour < session['close']
    
    async def get_market_volatility(self, pair: str, periods: int = 20) -> float:
        """Calculate the average true range (ATR) as a measure of volatility."""
        try:
            # Get recent candles (1-hour timeframe)
            engine = self.engines[pair]
            candles = await engine.get_candles(count=periods * 2, granularity='H1')
            
            if len(candles) < periods + 1:
                logger.warning(f"Not enough data to calculate volatility for {pair}")
                return 0.0
                
            # Calculate True Range (TR) for each period
            tr_values = []
            for i in range(1, len(candles)):
                prev_close = candles[i-1]['close']
                high = candles[i]['high']
                low = candles[i]['low']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            # Calculate ATR (simple moving average of TR)
            atr = sum(tr_values[-periods:]) / periods
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {pair}: {str(e)}")
            return 0.0
    
    async def is_pair_tradeable(self, pair: str) -> bool:
        """Check if a pair is tradeable based on market conditions."""
        # Check if market is open for this pair
        if not self.is_market_open(pair):
            logger.debug(f"Market closed for {pair}")
            return False
            
        # Check if we've reached the maximum trades for this pair today
        if self.today_trades.get(pair, 0) >= self.max_trades_per_pair:
            logger.debug(f"Maximum trades reached for {pair} today")
            return False
            
        # Check if we've reached the maximum number of open trades
        active_trades = len([t for t in self.active_trades.values() if t is not None])
        if active_trades >= self.max_open_trades:
            logger.debug("Maximum open trades reached")
            return False
            
        # Check if we already have an open position in this pair
        if self.active_trades.get(pair) is not None:
            logger.debug(f"Already have an open position in {pair}")
            return False
            
        # Check market volatility (optional)
        atr = await self.get_market_volatility(pair)
        if atr > 0.01:  # Example threshold, adjust based on pair and timeframe
            logger.debug(f"High volatility detected for {pair} (ATR: {atr:.5f})")
            # You might want to adjust position size based on volatility
            
        return True
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on the signal with enhanced validation."""
        pair = signal.get('pair')
        direction = signal.get('direction')
        confidence = signal.get('confidence', 0.5)
        
        # Validate input
        if not pair or direction not in ['buy', 'sell']:
            logger.error(f"Invalid trade signal: {signal}")
            return False
            
        if pair not in self.engines:
            logger.error(f"No engine found for pair {pair}")
            return False
            
        # Check if this pair is tradeable right now
        if not await self.is_pair_tradeable(pair):
            logger.debug(f"{pair} is not tradeable at this time")
            return False
            
        # Get the engine for this pair
        engine = self.engines[pair]
        
        try:
            # Get current price for better logging
            current_price = await engine.get_current_price()
            
            # Log the trade attempt with more details
            logger.info(f"Executing {direction.upper()} signal for {pair} at {current_price:.5f} "
                       f"with {confidence*100:.1f}% confidence")
            
            # Place the trade with additional context
            result = await engine.place_trade(
                direction, 
                reason=f"Signal: {signal.get('reason', 'No reason provided')}",
                confidence=confidence
            )
            
            if result:
                self.active_trades[pair] = result
                self.today_trades[pair] = self.today_trades.get(pair, 0) + 1
                
                # Log successful trade with details
                trade_id = result.get('id', 'unknown')
                units = result.get('units', 0)
                logger.info(f"Successfully executed {direction.upper()} trade for {units} units of {pair} "
                           f"(ID: {trade_id})")
                
                # Store trade details
                trade_details = {
                    'id': trade_id,
                    'pair': pair,
                    'direction': direction,
                    'entry_price': current_price,
                    'units': units,
                    'timestamp': datetime.utcnow().isoformat(),
                    'stop_loss': result.get('orderFillTransaction', {}).get('stopLossOnFill', {}).get('price'),
                    'take_profit': result.get('orderFillTransaction', {}).get('takeProfitOnFill', {}).get('price'),
                    'confidence': confidence
                }
                
                # Send Discord notification if configured
                if self.discord_webhook_url:
                    try:
                        await self.send_discord_notification(
                            f"✅ Trade Executed\n"
                            f"Pair: {pair}\n"
                            f"Direction: {direction.upper()}\n"
                            f"Price: {current_price:.5f}\n"
                            f"Units: {units}\n"
                            f"Confidence: {confidence*100:.1f}%"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send Discord notification: {str(e)}")
                
                return True
            else:
                logger.error(f"Failed to execute {direction.upper()} trade for {pair}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {str(e)}")
            logger.exception("Full traceback:")
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

async def main():
    """Main entry point for the OANDA Multi-Pair Scalper."""
    try:
        # Initialize the bot asynchronously
        logger.info("Initializing OANDA Multi-Pair Scalper...")
        bot = OandaMultiScalper()
        await bot.initialize()
        
        # Start the FastAPI server
        import uvicorn
        
        config = uvicorn.Config(
            app=bot.app,
            host="0.0.0.0",
            port=8003,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        logger.info("🚀 Starting OANDA Multi-Pair Scalper server...")
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        logger.exception("Stack trace:")
        raise

if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {str(e)}")
        raise
