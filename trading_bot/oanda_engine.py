import os
import logging
import asyncio
import pandas as pd
import numpy as np
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints import instruments, accounts
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.endpoints.accounts import AccountDetails
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import oandapyV20
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20.endpoints.accounts import AccountDetails, AccountSummary, AccountInstruments
from oandapyV20.endpoints.positions import OpenPositions, PositionDetails
from oandapyV20.endpoints.trades import OpenTrades, TradeDetails
from oandapyV20.endpoints.orders import OrderCreate, OrderCancel, OrderDetails
from oandapyV20.endpoints.pricing import PricingStream, PricingInfo
from oandapyV20.endpoints.instruments import InstrumentsCandles
from dotenv import load_dotenv
import json

try:
    from discord_webhook import DiscordWebhook
except ImportError:
    DiscordWebhook = None

# Import the forex pairs module
try:
    from .forex_pairs import get_top_forex_pairs
except ImportError:
    # Fallback in case of relative import issues
    from forex_pairs import get_top_forex_pairs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('oanda_engine.log')
    ]
)
logger = logging.getLogger('OandaEngine')

class OandaTradingEngine:
    """Trading engine for OANDA FX trading with 10% position sizing."""
    
    def __init__(self, config_file: str = '.env.oanda', discord_webhook_url: str = None, 
                 dry_run: bool = False, trading_pairs: List[str] = None):
        """
        Initialize the OandaTradingEngine.
        
        Args:
            config_file: Path to the configuration file
            discord_webhook_url: URL for Discord webhook notifications
            dry_run: If True, run in simulation mode without placing real trades
            trading_pairs: List of currency pairs to trade (e.g., ['EUR_USD', 'USD_JPY'])
        """
        # Initialize instance variables first to ensure they exist
        self.trading_pairs = []
        self.pair_states = {}
        self.client = None
        self.dry_run = dry_run
        self.discord_webhook = None
        self.account_id = None
        self.account_type = None
        self.account_currency = 'USD'
        self.risk_percent = 1.0
        self.stop_loss_pips = 20
        self.take_profit_pips = 40
        self.max_position_size = 1000000
        self.max_daily_trades = 10
        self.max_concurrent_pairs = 10
        self.trading_hours_start = 0
        self.trading_hours_end = 23
        self.open_positions = {}
        self.account_balance = 0.0
        self.unrealized_pnl = 0.0
        self.margin_available = 0.0
        self.trade_history = []
        self._trade_history_file = 'oanda_trade_history.json'
        
        # Load configuration with detailed logging
        config_path = os.path.abspath(config_file)
        logger.info(f"Loading configuration from: {config_path}")
        
        if not os.path.isfile(config_path):
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            # Check current working directory
            cwd = os.getcwd()
            logger.error(f"Current working directory: {cwd}")
            # List files in current directory
            try:
                files = os.listdir('.')
                logger.error(f"Files in current directory: {files}")
                env_files = [f for f in files if f.startswith('.env')]
                if env_files:
                    logger.error(f"Found these .env files: {env_files}")
            except Exception as e:
                logger.error(f"Error listing directory contents: {e}")
            raise FileNotFoundError(error_msg)
            
        # Load the environment variables
        logger.info("Loading environment variables...")
        load_dotenv(config_path, override=True)
        
        # Process trading pairs
        if trading_pairs is not None and isinstance(trading_pairs, list) and len(trading_pairs) > 0:
            self.trading_pairs = [p.upper().replace('-', '_') for p in trading_pairs]
        elif 'TRADING_PAIRS' in os.environ:
            self.trading_pairs = [p.strip().upper().replace('-', '_') 
                                for p in os.environ['TRADING_PAIRS'].split(',') 
                                if p.strip()]
        else:
            # Default to EUR/USD if no pairs specified
            self.trading_pairs = ['EUR_USD']
        
        # Initialize pair states
        self.pair_states = {
            pair: {
                'last_signal': None,
                'last_price': None,
                'position_size': 0,
                'open_trades': 0,
                'daily_trades': 0,
                'last_trade_time': None,
                'indicators': {}
            } for pair in self.trading_pairs
        }
        
        # Log important configuration values (masking sensitive data)
        config_vars = {
            'OANDA_ACCOUNT_ID': os.getenv('OANDA_ACCOUNT_ID', 'NOT SET'),
            'OANDA_ACCOUNT_TYPE': os.getenv('OANDA_ACCOUNT_TYPE', 'practice'),
            'TRADING_PAIR': os.getenv('TRADING_PAIR', 'EUR_USD'),
            'DRY_RUN': os.getenv('DRY_RUN', 'false'),
            'HEADLESS': os.getenv('HEADLESS', 'false'),
            'MAX_CONCURRENT_PAIRS': os.getenv('MAX_CONCURRENT_PAIRS', '10')
        }
        
        # Log the configuration (masking sensitive data)
        masked_config = config_vars.copy()
        if masked_config['OANDA_ACCOUNT_ID'] != 'NOT SET':
            masked_config['OANDA_ACCOUNT_ID'] = f"{masked_config['OANDA_ACCOUNT_ID'][:3]}...{masked_config['OANDA_ACCOUNT_ID'][-3:]}"
            
        logger.info(f"Configuration loaded: {masked_config}")
        
        # Initialize OANDA API client with enhanced error handling
        try:
            api_key = os.getenv('OANDA_API_KEY')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID')
            
            if not api_key or not self.account_id:
                raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in .env.oanda")
                
            environment = os.getenv('OANDA_ACCOUNT_TYPE', 'practice').lower()
            logger.info(f"Initializing OANDA API client for {environment.upper()} environment")
            logger.debug(f"Using Account ID: {self.account_id[:5]}...{self.account_id[-3:]}")
            
            # Store API key and account type as instance variables
            self.api_key = api_key
            self.account_type = environment  # 'practice' or 'live'
            
            # Initialize the API client
            self.client = oandapyV20.API(
                access_token=self.api_key,
                environment=self.account_type
            )
            
            # Test the connection using AccountDetails endpoint
            try:
                account_details = AccountDetails(accountID=self.account_id)
                response = self.client.request(account_details)
                
                if 'account' in response:
                    account_info = response['account']
                    logger.info(f"Successfully connected to OANDA account: {account_info.get('alias', 'N/A')}")
                    logger.info(f"Account Balance: {account_info.get('balance', 'N/A')} {account_info.get('currency', 'USD')}")
                    logger.info(f"Margin Available: {account_info.get('marginAvailable', 'N/A')} {account_info.get('currency', 'USD')}")
                    
                    # Store account information
                    self.account_balance = float(account_info.get('balance', 0))
                    self.margin_available = float(account_info.get('marginAvailable', 0))
                    self.unrealized_pnl = float(account_info.get('unrealizedPL', 0))
                    self.account_currency = account_info.get('currency', 'USD')
                    
                else:
                    logger.warning("Unexpected account details format in API response")
                    
            except V20Error as e:
                logger.error(f"OANDA API Error: {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    logger.error(f"API Response: {e.response.text}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize OANDA API client: {str(e)}", exc_info=True)
            raise
        # Initialize trading parameters
        self.dry_run = dry_run or os.getenv('DRY_RUN', 'false').lower() == 'true'
        
        # Validate required configuration
        self.api_key = os.getenv('OANDA_API_KEY')
        if not self.api_key:
            raise ValueError("OANDA_API_KEY not found in environment variables")
            
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
            
        self.account_type = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')
        if self.account_type not in ['practice', 'live']:
            raise ValueError("OANDA_ACCOUNT_TYPE must be either 'practice' or 'live'")
            
        # Initialize the OANDA API client
        self.client = API(access_token=self.api_key, environment=self.account_type)
        logger.info(f"Successfully initialized OANDA API client for {self.account_type.upper()} environment")
            
        # Initialize trading parameters
        if dry_run:
            logger.warning("DRY RUN MODE ENABLED - No actual trades will be placed")
        
        # If no pairs were provided or found, use the top 25 forex pairs
        if not trading_pairs:
            logger.info("No trading pairs specified, using top 25 forex pairs by volume")
            self.trading_pairs = get_top_forex_pairs()
        else:
            self.trading_pairs = [p.replace('-', '_').upper() for p in trading_pairs]
        
        logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")
        
        self.account_currency = os.getenv('ACCOUNT_CURRENCY', 'USD')
        self.risk_percent = float(os.getenv('RISK_PERCENT', '1.0'))  # 1% risk per trade
        self.stop_loss_pips = float(os.getenv('STOP_LOSS_PIPS', '20'))  # 20 pips stop loss
        self.take_profit_pips = float(os.getenv('TAKE_PROFIT_PIPS', '40'))  # 40 pips take profit
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '1000000'))  # Max position size in units
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '10'))  # Max trades per day
        self.max_concurrent_pairs = int(os.getenv('MAX_CONCURRENT_PAIRS', '10'))  # Max pairs to trade simultaneously
        self.trading_hours_start = int(os.getenv('TRADING_HOURS_START', '0'))  # 12:00 AM UTC
        self.trading_hours_end = int(os.getenv('TRADING_HOURS_END', '23'))  # 11:00 PM UTC
        
        # Track state for each trading pair
        self.pair_states = {pair: {
            'last_signal': None,
            'last_price': None,
            'position_size': 0,
            'open_trades': 0,
            'daily_trades': 0,
            'last_trade_time': None,
            'indicators': {}
        } for pair in self.trading_pairs}
        
        # Discord webhook for notifications
        self.discord_webhook = None
        if discord_webhook_url and DiscordWebhook is not None:
            self.discord_webhook = DiscordWebhook(discord_webhook_url)
            logger.info("Discord webhook initialized")
        
        # State
        self.open_positions = {}
        self.account_balance = 0.0
        self.unrealized_pnl = 0.0
        self.margin_available = 0.0
        
        # For backward compatibility, set the first trading pair as the default
        self.trading_pair = self.trading_pairs[0] if self.trading_pairs else None
        
        logger.info(f"Initialized OANDA Trading Engine for {len(self.trading_pairs)} pairs: {', '.join(self.trading_pairs[:5])}" + 
                   ("..." if len(self.trading_pairs) > 5 else ""))
        logger.info(f"Risk Settings: {self.risk_percent*100}% risk, SL: {self.stop_loss_pips} pips, "
                   f"TP: {self.take_profit_pips} pips, Max Daily Trades: {self.max_daily_trades}")
        
        # Initialize trade history
        self.trade_history = []
        self._trade_history_file = 'oanda_trade_history.json'
        self._load_trade_history()
    
    async def get_account_balance(self) -> float:
        """Get the current account balance with improved error handling and validation."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get account details
                r = AccountDetails(accountID=self.account_id)
                response = self.client.request(r)
                
                if not response or 'account' not in response:
                    raise ValueError("Invalid response from OANDA API")
                
                # Update account metrics
                account = response.get('account', {})
                self.account_balance = float(account.get('balance', 0.0))
                self.unrealized_pnl = float(account.get('unrealizedPL', 0.0))
                self.margin_available = float(account.get('marginAvailable', 0.0))
                
                # Log account information
                logger.info(f"Account: {self.account_id}")
                logger.info(f"Balance: {self.account_balance} {self.account_currency}")
                logger.info(f"Unrealized P/L: {self.unrealized_pnl} {self.account_currency}")
                logger.info(f"Margin Available: {self.margin_available} {self.account_currency}")
                
                return self.account_balance
                
            except V20Error as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt + 1}/{max_retries} - Error getting account balance: {error_msg}")
                
                # Check if this is a rate limit error
                if '429' in error_msg and attempt < max_retries - 1:
                    logger.warning(f"Rate limited. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                    
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"API Response: {e.response.text}")
                
                if attempt == max_retries - 1:  # Last attempt
                    logger.error("Max retries reached. Could not get account balance.")
                    return 0.0
                    
            except Exception as e:
                logger.error(f"Unexpected error in get_account_balance: {str(e)}")
                logger.exception("Full traceback:")
                if attempt == max_retries - 1:  # Last attempt
                    return 0.0
                    
            # Small delay before next retry
            await asyncio.sleep(1)
        
        return 0.0
    
    def get_position_size(self, entry_price: float, stop_loss_price: float) -> Tuple[float, float]:
        """
        Calculate position size based on 10% of account balance, stop loss, and available margin.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            
        Returns:
            Tuple[units, position_value] - Number of units to trade and their value in account currency
        """
        try:
            # Get current account balance and margin available
            asyncio.run(self.get_account_balance())
            
            if self.account_balance <= 0:
                logger.error("Invalid account balance")
                return 0.0, 0.0
                
            # Calculate risk amount (10% of account balance)
            risk_amount = min(self.account_balance * self.risk_percent, self.margin_available)
            
            if risk_amount <= 0:
                logger.error(f"Insufficient margin available. Margin: {self.margin_available}")
                return 0.0, 0.0
            
            # Calculate pip value (assuming standard lot size of 100,000 units)
            # For pairs where USD is the quote currency, 1 pip = 0.0001
            pip_value = 0.0001 * 100000  # Value of 1 pip per standard lot
            
            # Calculate position size in units
            stop_loss_distance = abs(entry_price - stop_loss_price)
            pips_risk = stop_loss_distance / 0.0001  # Convert price difference to pips
            
            if pips_risk == 0:
                logger.warning("Stop loss is too close to entry price")
                return 0.0, 0.0
                
            # Calculate position size in account currency
            position_value = (risk_amount * 100000) / pips_risk
            
            # Convert to units (1 standard lot = 100,000 units)
            units = (position_value / entry_price) * 100000
            
            # Get maximum allowed position size from OANDA
            max_units = self.get_max_position_size()
            
            # Ensure we don't exceed maximum position size
            if max_units > 0:
                units = min(units, max_units)
            
            # Ensure we don't exceed available margin (safety check)
            margin_required = (units * entry_price) / 50  # 50:1 leverage
            if margin_required > self.margin_available:
                units = (self.margin_available * 50) / entry_price
            
            # Round to nearest 1000 units (OANDA requirement)
            units = round(units / 1000) * 1000
            
            # Recalculate position value based on final units
            position_value = (units * entry_price) / 100000
            
            # Final safety check to ensure we don't exceed max units
            units = min(units, max_units) if max_units > 0 else units
            
            logger.info(f"Position size: {units:,} units (${position_value:,.2f} at {entry_price:.5f})")
            logger.info(f"Risk: {self.risk_percent*100:.1f}% of ${self.account_balance:,.2f} = ${risk_amount:,.2f}")
            
            return units, position_value
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0, 0.0
            
    def get_max_position_size(self) -> float:
        """
        Get the maximum position size allowed by OANDA based on account type and instrument.
        
        Returns:
            float: Maximum number of units allowed for a single position
        """
        # OANDA standard position size limits (in units)
        limits = {
            'standard': {
                'majors': 10_000_000,     # EUR/USD, GBP/USD, etc.
                'minors': 5_000_000,      # EUR/GBP, AUD/CAD, etc.
                'exotics': 1_000_000,     # USD/MXN, USD/TRY, etc.
                'jpy_pairs': 10_000_000,  # USD/JPY, EUR/JPY, etc.
            },
            'premium': {
                'majors': 50_000_000,
                'minors': 25_000_000,
                'exotics': 5_000_000,
                'jpy_pairs': 50_000_000,
            }
        }
        
        # Determine account type (standard or premium)
        account_type = 'premium' if self.account_type.lower() == 'live' else 'standard'
        
        # Categorize the trading pair
        majors = ['EUR_USD', 'GBP_USD', 'USD_CHF', 'AUD_USD', 'NZD_USD', 'USD_CAD']
        jpy_pairs = ['USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY']
        
        if self.trading_pair in majors:
            pair_type = 'majors'
        elif self.trading_pair in jpy_pairs:
            pair_type = 'jpy_pairs'
        elif any(c in self.trading_pair for c in ['MXN', 'TRY', 'ZAR', 'NOK', 'SEK', 'HKD', 'SGD']):
            pair_type = 'exotics'
        else:
            pair_type = 'minors'
        
        # Get the maximum units allowed
        max_units = limits[account_type][pair_type]
        
        # Additional safety margin (90% of limit to be safe)
        max_units = int(max_units * 0.9)
        
        logger.info(f"Max position size for {self.trading_pair} ({pair_type}): {max_units:,} units")
        return max_units
    
    async def get_current_price(self) -> float:
        """Get the current price of the trading pair."""
        try:
            from oandapyV20.endpoints.pricing import PricingInfo
            
            params = {
                'instruments': self.trading_pair
            }
            
            # Get the pricing information
            r = PricingInfo(accountID=self.account_id, params=params)
            response = self.client.request(r)
            
            # Extract the bid price
            if 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]
                if 'bids' in price_data and len(price_data['bids']) > 0:
                    return float(price_data['bids'][0]['price'])
            
            logger.warning("No price data available in response")
            raise ValueError("No price data available in response")
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            raise
            
        except V20Error as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    async def get_historical_data(self, pair: str = None, count: int = 100, granularity: str = "M5") -> Optional[pd.DataFrame]:
        """
        Get historical price data for a specific currency pair.
        
        Args:
            pair: The currency pair to get data for (e.g., 'EUR_USD').
                  If None, uses the first pair in self.trading_pairs.
            count: Number of candles to retrieve.
            granularity: The candle granularity (e.g., 'M5' for 5 minutes).
            
        Returns:
            DataFrame with OHLCV data or None if an error occurs.
        """
        try:
            # Use provided pair or default to the first trading pair
            instrument = pair if pair is not None else self.trading_pairs[0]
            
            # Convert pair format from EUR/USD to EUR_USD if needed
            if '/' in instrument:
                instrument = instrument.replace('/', '_')
            
            # Use the Candles endpoint directly
            params = {
                "count": count,
                "granularity": granularity,
                "price": "MBA"  # Mid, Bid, Ask
            }
            
            # Create request
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            
            # Make the request
            response = self.client.request(request)
            
            # Process the response
            candles = []
            for candle in response.get('candles', []):
                if 'mid' in candle and candle['mid']:
                    candles.append({
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            # Convert to DataFrame
            if not candles:
                logger.warning(f"No candle data received for {instrument}")
                return None
                
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            return df
            
        except V20Error as e:
            logger.error(f"OANDA API error getting historical data for {instrument}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting historical data for {instrument}: {e}", exc_info=True)
            return None
    
    def _load_trade_history(self):
        """Load trade history from file if it exists"""
        try:
            if os.path.exists(self._trade_history_file):
                with open(self._trade_history_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trade_history = []
    
    def _save_trade_history(self):
        """Save trade history to file"""
        try:
            with open(self._trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info(f"Trade history saved to {self._trade_history_file}")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
            
    async def close(self):
        """
        Close the trading engine and clean up resources.
        """
        try:
            logger.info("Closing OandaTradingEngine...")
            # Save any pending data
            self._save_trade_history()
            
            # Close any open connections
            if hasattr(self, 'client') and self.client is not None:
                # OANDA's API client doesn't have a close method, but we'll log it
                logger.info("OANDA API client closed")
                self.client = None
                
            logger.info("OandaTradingEngine closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing OandaTradingEngine: {e}", exc_info=True)
            raise
    
    async def _confirm_trade(self, trade_details: Dict) -> bool:
        """
        Request confirmation for a trade before execution.
        
        Args:
            trade_details: Dictionary containing trade details
            
        Returns:
            bool: True if trade is confirmed, False otherwise
        """
        # In dry-run mode, always confirm
        if self.dry_run:
            logger.warning("DRY RUN: Trade would be executed in live mode")
            return True
            
        # Log trade details for review
        logger.info("\n" + "="*80)
        logger.info(f"📊 TRADE SIGNAL DETECTED - {trade_details['direction'].upper()} {self.trading_pair}")
        logger.info("-" * 40)
        logger.info(f"🔹 Entry Price: {trade_details['entry_price']:.5f}")
        logger.info(f"🔹 Stop Loss: {trade_details['stop_loss']:.5f} ({trade_details['stop_loss_pips']} pips)")
        logger.info(f"🔹 Take Profit: {trade_details['take_profit']:.5f} ({trade_details['take_profit_pips']} pips)")
        logger.info(f"🔹 Position Size: {trade_details['units']:,} units")
        logger.info(f"🔹 Risk: {trade_details['risk_amount']:.2f} {self.account_currency} ({self.risk_percent*100}% of balance)")
        logger.info(f"🔹 Reason: {trade_details.get('reason', 'No reason provided')}")
        logger.info("="*80 + "\n")
        
        # In headless mode, auto-confirm if not in dry-run
        if os.getenv('HEADLESS', 'false').lower() == 'true':
            logger.info("HEADLESS MODE: Auto-confirming trade")
            return True
            
        # Otherwise, prompt for confirmation
        try:
            confirm = input("\nExecute this trade? (y/n): ").strip().lower()
            return confirm == 'y'
        except Exception as e:
            logger.error(f"Error getting trade confirmation: {e}")
            return False
    
    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit has been reached"""
        today = datetime.utcnow().date()
        
        # Reset counter if it's a new day
        if self.last_trade_day != today:
            self.daily_trade_count = 0
            self.last_trade_day = today
            
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"Daily trade limit of {self.max_daily_trades} reached")
            return False
            
        return True
    
    async def place_trade(self, direction: str, entry_price: float = None, reason: str = None) -> Optional[Dict]:
        """
        Place a trade with proper position sizing, risk management, and safety checks.
        
        Args:
            direction: 'buy' or 'sell'
            entry_price: Optional entry price (if None, uses current market price)
            reason: Reason for the trade (for logging/notifications)
            
        Returns:
            Trade details or None if failed
        """
        start_time = time.time()
        trade_id = str(uuid.uuid4())[:8]
        
        # Initialize response object
        response = {
            'success': False,
            'message': '',
            'trade_id': trade_id,
            'dry_run': self.dry_run,
            'execution_time': 0,
            'details': {}
        }
        
        try:
            # 1. Validate direction
            direction = direction.lower()
            if direction not in ['buy', 'sell']:
                raise ValueError("Direction must be 'buy' or 'sell'")
            
            # 2. Check daily trade limit
            if not self._check_daily_trade_limit():
                raise ValueError(f"Daily trade limit of {self.max_daily_trades} reached")
            
            # 3. Get the latest account balance and margin information
            account_balance = await self.get_account_balance()
            if account_balance <= 0:
                error_msg = "Invalid account balance. Cannot place trade."
                logger.error(error_msg)
                response['message'] = error_msg
                return response
            
            # 4. Get current price if not provided
            current_price = await self.get_current_price()
            if current_price is None:
                error_msg = "Could not get current price"
                logger.error(error_msg)
                response['message'] = error_msg
                return response
                
            entry_price = entry_price or current_price
            
            # 5. Calculate stop loss and take profit prices with safety limits
            stop_loss_pips = self.stop_loss_pips
            take_profit_pips = self.take_profit_pips
            
            # Calculate pip value based on currency pair
            pip_multiplier = 0.0001  # For most currency pairs
            if 'JPY' in self.trading_pair:
                pip_multiplier = 0.01  # For JPY pairs
                
            # Calculate stop loss and take profit prices
            if direction == 'buy':
                stop_price = round(entry_price - (stop_loss_pips * pip_multiplier), 5)
                take_profit_price = round(entry_price + (take_profit_pips * pip_multiplier), 5)
            else:  # sell
                stop_price = round(entry_price + (stop_loss_pips * pip_multiplier), 5)
                take_profit_price = round(entry_price - (take_profit_pips * pip_multiplier), 5)
            
            # 6. Calculate position size with risk management
            risk_amount = account_balance * self.risk_percent
            risk_per_unit = abs(entry_price - stop_price) * 100000  # For standard lot (100,000 units)
            
            if risk_per_unit == 0:
                raise ValueError("Invalid stop loss - too close to entry price")
                
            # Calculate base position size
            position_size = int((risk_amount / risk_per_unit) * 100000)
            
            # Apply position size limits
            max_allowed = self.get_max_position_size()
            position_size = max(1000, min(position_size, max_allowed))  # Min 1000 units, max allowed
            
            # Round to nearest 1000 units (OANDA requirement)
            position_size = round(position_size / 1000) * 1000
            
            # 7. Prepare trade details for confirmation
            trade_details = {
                'id': trade_id,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_price,
                'take_profit': take_profit_price,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'units': position_size,
                'risk_amount': risk_amount,
                'account_balance': account_balance,
                'margin_available': self.margin_available,
                'unrealized_pnl': self.unrealized_pnl,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': reason or 'No reason provided'
            }
            
            # 8. Request trade confirmation
            if not await self._confirm_trade(trade_details):
                response.update({
                    'success': False,
                    'message': 'Trade cancelled by user',
                    'details': trade_details
                })
                return response
            
            # 9. Execute the trade (if not in dry-run mode)
            if self.dry_run:
                logger.warning(f"DRY RUN: Would place {direction.upper()} order for {position_size:,} units of {self.trading_pair}")
                response.update({
                    'success': True,
                    'message': 'Dry run - no trade executed',
                    'details': trade_details
                })
                return response
            
            # 10. Place the actual trade with OANDA
            logger.info(f"Executing {direction.upper()} order for {position_size:,} units of {self.trading_pair}")
            
            # Prepare the order request
            order_data = {
                "order": {
                    "units": str(position_size) if direction == 'buy' else str(-position_size),
                    "instrument": self.trading_pair,
                    "timeInForce": "FOK",  # Fill or Kill
                    "type": "MARKET",
                    "stopLossOnFill": {
                        "timeInForce": "GTC",  # Good Till Cancelled
                        "price": str(stop_price)
                    },
                    "takeProfitOnFill": {
                        "timeInForce": "GTC",
                        "price": str(take_profit_price)
                    },
                    "positionFill": "DEFAULT"
                }
            }
            
            # Execute the order
            api_response = self.client.order.create(self.account_id, order_data)
            
            # Update trade details with execution info
            trade_details.update({
                'execution_time': time.time() - start_time,
                'api_response': api_response,
                'order_id': api_response.get('orderFillTransaction', {}).get('id', 'unknown')
            })
            
            # Update trade history and daily count
            self.trade_history.append(trade_details)
            self._save_trade_history()
            self.daily_trade_count += 1
            
            # Log successful execution
            logger.info(f"✅ Trade executed successfully in {trade_details['execution_time']:.2f}s")
            
            # Send Discord notification
            await self._send_trade_notification(trade_details, success=True)
            
            # Prepare success response
            response.update({
                'success': True,
                'message': 'Trade executed successfully',
                'details': trade_details,
                'execution_time': trade_details['execution_time']
            })
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error executing {direction.upper()} trade: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error notification
            await self._send_trade_notification({
                'id': trade_id,
                'pair': self.trading_pair,
                'direction': direction,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, success=False)
            
            response.update({
                'message': error_msg,
                'error': str(e),
                'execution_time': time.time() - start_time
            })
            return response
    
    async def _send_trade_notification(self, trade_data: Dict, success: bool):
        """Send trade notification to Discord"""
        if not self.discord_webhook:
            return
            
        try:
            color = 0x00ff00 if success else 0xff0000
            title = f"✅ Trade Executed" if success else f"❌ Trade Error"
            
            embed = {
                "title": f"{title}: {trade_data.get('direction', '').upper()} {trade_data.get('pair', '')}",
                "color": color,
                "timestamp": trade_data.get('timestamp', datetime.utcnow().isoformat()),
                "fields": []
            }
            
            if success:
                embed["fields"].extend([
                    {"name": "Entry Price", "value": f"{trade_data.get('entry_price', 0):.5f}", "inline": True},
                    {"name": "Stop Loss", "value": f"{trade_data.get('stop_loss', 0):.5f} ({trade_data.get('stop_loss_pips', 0)} pips)", "inline": True},
                    {"name": "Take Profit", "value": f"{trade_data.get('take_profit', 0):.5f} ({trade_data.get('take_profit_pips', 0)} pips)", "inline": True},
                    {"name": "Position Size", "value": f"{trade_data.get('units', 0):,} units", "inline": True},
                    {"name": "Risk Amount", "value": f"{trade_data.get('risk_amount', 0):.2f} {self.account_currency}", "inline": True},
                    {"name": "Execution Time", "value": f"{trade_data.get('execution_time', 0):.2f}s", "inline": True},
                    {"name": "Reason", "value": trade_data.get('reason', 'No reason provided'), "inline": False}
                ])
            else:
                embed["description"] = f"**Error:** {trade_data.get('error', 'Unknown error')}"
            
            # Clone the webhook to avoid threading issues
            webhook = DiscordWebhook(url=self.discord_webhook.url)
            webhook.add_embed(embed)
            webhook.execute()
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {str(e)}")
                
            if direction.lower() == 'buy':
                stop_loss_price = round(entry_price - (self.stop_loss_pips * pip_multiplier), 5)
                take_profit_price = round(entry_price + (self.take_profit_pips * pip_multiplier), 5)
                units = await self.calculate_position_size(entry_price, stop_loss_price)
            else:  # sell
                stop_loss_price = round(entry_price + (self.stop_loss_pips * pip_multiplier), 5)
                take_profit_price = round(entry_price - (self.take_profit_pips * pip_multiplier), 5)
                units = -await self.calculate_position_size(entry_price, stop_loss_price)
            
            if units == 0:
                error_msg = "Invalid position size"
                logger.error(error_msg)
                if self.discord_webhook:
                    self.discord_webhook.send_error(error_msg)
                return None
            
            # Prepare order data
            order_data = {
                "order": {
                    "units": str(int(units)),
                    "instrument": self.trading_pair,
                    "timeInForce": "FOK",  # Fill or Kill
                    "type": "MARKET",
                    "stopLossOnFill": {
                        "price": f"{stop_loss_price:.5f}",
                        "timeInForce": "GTC"  # Good Till Cancelled
                    },
                    "takeProfitOnFill": {
                        "price": f"{take_profit_price:.5f}",
                        "timeInForce": "GTC"
                    },
                    "positionFill": "DEFAULT"
                }
            }
            
            # Log trade details
            trade_info = (
                f"Placing {direction.upper()} order for {self.trading_pair}:\n"
                f"- Entry: {entry_price:.5f}\n"
                f"- Stop Loss: {stop_loss_price:.5f}\n"
                f"- Take Profit: {take_profit_price:.5f}\n"
                f"- Units: {int(units)}"
            )
            
            if reason:
                trade_info += f"\n- Reason: {reason}"
                
            logger.info(trade_info)
            
            # Place the order
            try:
                r = OrderCreate(accountID=self.account_id, data=order_data)
                response = self.client.request(r)
                logger.debug(f"OANDA API Response: {response}")
                
                if 'orderFillTransaction' in response:
                    logger.info(f"Successfully placed {direction.upper()} order")
                    
                    # Get the trade ID if available
                    trade_id = None
                    if 'tradeOpened' in response.get('orderFillTransaction', {}):
                        trade_id = response['orderFillTransaction']['tradeOpened'].get('tradeID')
                    
                    # Send success notification to Discord
                    if self.discord_webhook:
                        self.discord_webhook.send_trade_opened(
                            pair=self.trading_pair,
                            direction=direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price,
                            units=abs(units)
                        )
                    
                    logger.info(f"Trade executed successfully. Trade ID: {trade_id}")
                    return {
                        'id': trade_id,
                        'instrument': self.trading_pair,
                        'direction': direction,
                        'units': abs(units),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'timestamp': datetime.utcnow().isoformat(),
                        'raw_response': response
                    }
                else:
                    error_msg = "Unexpected response format from OANDA API"
                    logger.error(f"{error_msg}: {response}")
                    if self.discord_webhook:
                        self.discord_webhook.send_error(error_msg)
                    return None
                
            except V20Error as e:
                error_msg = f"OANDA API Error: {e}"
                if hasattr(e, 'response') and e.response is not None:
                    error_msg += f"\nAPI Response: {e.response.text}"
                    logger.error(f"OANDA API Error Response: {e.response.text}")
                
                logger.error(error_msg, exc_info=True)
                
                # Send error notification to Discord
                if self.discord_webhook:
                    self.discord_webhook.send_error(f"Failed to place order: {error_msg}")
                
                return None
                
        except Exception as e:
            error_msg = f"Unexpected error in place_trade: {e}"
            logger.error(error_msg, exc_info=True)
            
            if self.discord_webhook:
                self.discord_webhook.send_error(error_msg)
                
            return None
            
    async def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate position size starting from 250 units and scaling with account size.
        Always uses the latest account balance and includes safety checks.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            
        Returns:
            float: Number of units to trade, or 0 if invalid
        """
        try:
            # Always get the latest account balance and margin information
            current_balance = await self.get_account_balance()
            
            # Validate account balance
            if current_balance <= 0:
                logger.error(f"Invalid account balance: {current_balance:.2f}")
                return 0.0
                
            # Log current account status for debugging
            logger.info(
                f"Position Sizing - Balance: {self.account_balance:.2f} {self.account_currency}, "
                f"Margin Available: {self.margin_available:.2f}, "
                f"Unrealized P&L: {self.unrealized_pnl:.2f}"
            )
            
            # Base position size (starting point)
            base_units = 250.0
            
            # Scale position size based on account balance
            # For every $1000 in account balance, add 100 units
            balance_scaling = max(1.0, current_balance / 1000.0)
            scaled_units = base_units * balance_scaling
            
            # Ensure position size is a multiple of 1000 (OANDA requirement)
            units = round(scaled_units / 1000) * 1000
            
            # Ensure minimum position size of 250 units
            units = max(units, 250)
            
            # Calculate required margin and check against available margin
            margin_required = (units * entry_price) / 50  # Assuming 50:1 leverage
            
            # Add a safety buffer (use only 80% of available margin)
            safe_margin_available = self.margin_available * 0.8
            
            if margin_required > safe_margin_available:
                # Recalculate units based on safe margin
                units = int((safe_margin_available * 50) / entry_price)
                units = max(units, 250)  # Maintain minimum size
                logger.info(f"Adjusted position size for margin safety: {units} units")
            
            # Get maximum allowed position size
            max_units = self.get_max_position_size()
            if max_units > 0:
                units = min(units, max_units)
                
                if units < 250:
                    logger.warning("Calculated position size is below minimum (250 units)")
                    return 0.0
            
            # Final validation
            if units <= 0:
                logger.error(f"Invalid position size calculated: {units}")
                return 0.0
                
            # Log the position size calculation
            position_value = (units * entry_price) / 100000  # Position value in lots
            logger.info(
                f"Position Size - Units: {units:,}, "
                f"Value: {position_value:.2f} lots, "
                f"Margin Req: ${(units * entry_price) / 50:,.2f}"
            )
                
            return float(units)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a price series.
        
        Args:
            prices: Pandas Series of price data
            period: Number of periods to use for RSI calculation (default: 14)
            
        Returns:
            Pandas Series containing RSI values
        """
        try:
            delta = prices.diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    async def generate_signal(self, pair: str = None) -> Optional[str]:
        """
        Generate a trading signal for the specified currency pair.
        
        Args:
            pair: The currency pair to generate a signal for (e.g., 'EUR_USD').
                  If None, uses the first pair in self.trading_pairs.
                  
        Returns:
            'BUY', 'SELL', or 'HOLD' based on the trading strategy, or None on error.
        """
        # Use provided pair or default to the first trading pair
        pair = pair or (self.trading_pairs[0] if self.trading_pairs else None)
        if not pair:
            logger.error("No trading pair specified and no default pairs available")
            return None
            
        # Check if we have recent signals for this pair to prevent spamming
        current_time = datetime.utcnow()
        if pair in self.pair_states:
            last_signal_time = self.pair_states[pair].get('last_signal_time')
            if last_signal_time and (current_time - last_signal_time).total_seconds() < 300:  # 5 minutes cooldown
                logger.debug(f"Skipping signal generation for {pair} - too soon after last signal")
                return None
            
        try:
            # Get historical data for analysis
            df = await self.get_historical_data(pair=pair, count=100, granularity="M5")
            if df is None or df.empty:
                logger.error(f"No historical data available for {pair}")
                return None
                
            # Calculate technical indicators
            df['rsi'] = self._calculate_rsi(df['close'], period=14)
            
            # Get the latest values
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize signal state for this pair if it doesn't exist
            if pair not in self.pair_states:
                self.pair_states[pair] = {
                    'last_signal': None,
                    'last_signal_time': None,
                    'last_price': None,
                    'rsi': None,
                    'consecutive_signals': 0
                }
            
            # Get current signal state
            signal_state = self.pair_states[pair]
            
            # Update pair state with latest values
            signal_state.update({
                'last_price': float(current['close']),
                'rsi': float(current['rsi']) if not pd.isna(current['rsi']) else None
            })
            
            # Check if we have valid RSI data
            if pd.isna(current['rsi']) or pd.isna(prev['rsi']):
                logger.warning(f"Insufficient data for RSI calculation on {pair}")
                return None
            
            # Generate signal based on RSI and price action with confirmation
            signal = None
            
            # Check for oversold condition with confirmation (potential buy)
            if (current['rsi'] < 30 and 
                prev['rsi'] < current['rsi'] and 
                current['close'] > prev['close']):
                signal = 'BUY'
            # Check for overbought condition with confirmation (potential sell)
            elif (current['rsi'] > 70 and 
                  prev['rsi'] > current['rsi'] and 
                  current['close'] < prev['close']):
                signal = 'SELL'
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating trading signal for {pair}: {str(e)}", exc_info=True)
            return None
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            r = OpenPositions(accountID=self.account_id)
            response = self.client.request(r)
            return response.get('positions', [])
        except V20Error as e:
            logger.error(f"Error getting open positions: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API Response: {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_open_positions: {e}")
            return []
    
    async def close_position(self, instrument: str, long_units: float = 0.0, short_units: float = 0.0) -> bool:
        """Close a specific position."""
        try:
            data = {}
            if long_units > 0:
                data["longUnits"] = "ALL"
            if short_units < 0:
                data["shortUnits"] = "ALL"
                
            if not data:
                return True  # Nothing to close
                
            r = PositionDetails(accountID=self.account_id, instrument=instrument, data=data)
            self.client.request(r)
            logger.info(f"Closed position for {instrument}: {data}")
            return True
            
        except V20Error as e:
            logger.error(f"Error closing position for {instrument}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API Response: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error closing position for {instrument}: {e}")
            return False
    
    async def close_all_positions(self) -> bool:
        """Close all open positions."""
        success = True
        try:
            positions = await self.get_open_positions()
            
            for position in positions:
                instrument = position.get('instrument')
                long_units = float(position.get('long', {}).get('units', 0))
                short_units = float(position.get('short', {}).get('units', 0))
                
                if long_units != 0 or short_units != 0:
                    if not await self.close_position(instrument, long_units, short_units):
                        success = False
                        
            if success:
                logger.info("All positions closed successfully")
            else:
                logger.warning("Some positions may not have been closed successfully")
                
            return success
            
        except Exception as e:
            logger.error(f"Error in close_all_positions: {e}")
            return False
    
    async def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        try:
            r = OpenTrades(accountID=self.account_id)
            response = self.client.request(r)
            return response.get('trades', [])
        except V20Error as e:
            logger.error(f"Error getting open trades: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API Response: {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_open_trades: {e}")
            return []
    
    async def get_current_price(self, pair: str = None) -> Optional[float]:
        """
        Get the current market price for the specified trading pair.
        
        Args:
            pair: The currency pair to get the price for (e.g., 'EUR_USD'). 
                  If None, uses the first pair in self.trading_pairs.
                  
        Returns:
            The current mid price as a float, or None if the price couldn't be retrieved.
        """
        if pair is None:
            if not self.trading_pairs:
                logger.error("No trading pairs available")
                return None
            pair = self.trading_pairs[0]
            
        try:
            # First try to get price from the instruments candles endpoint
            from oandapyV20.endpoints.instruments import InstrumentsCandles
            
            # Get the most recent candle
            params = {
                'count': 1,
                'granularity': 'M1',  # 1-minute candles
                'price': 'M'          # Midpoint (average of bid/ask)
            }
            
            try:
                candles = InstrumentsCandles(
                    instrument=pair,
                    params=params
                )
                
                response = self.client.request(candles)
                
                if 'candles' in response and response['candles']:
                    # Return the close price of the most recent candle
                    price = float(response['candles'][0]['mid']['c'])
                    self.pair_states[pair]['last_price'] = price
                    return price
                    
            except V20Error as e:
                logger.warning(f"Candles endpoint failed for {pair}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error with candles endpoint for {pair}: {str(e)}")
            
            # Fallback to using the pricing endpoint if candles endpoint fails
            try:
                from oandapyV20.endpoints.pricing import PricingInfo
                
                params = {
                    'instruments': pair
                }
                
                pricing = PricingInfo(
                    accountID=self.account_id,
                    params=params
                )
                
                response = self.client.request(pricing)
                
                if 'prices' in response and response['prices']:
                    # Return the mid price (average of bid/ask)
                    bid = float(response['prices'][0]['bids'][0]['price'])
                    ask = float(response['prices'][0]['asks'][0]['price'])
                    price = (bid + ask) / 2
                    self.pair_states[pair]['last_price'] = price
                    return price
                    
            except V20Error as e:
                logger.warning(f"Pricing endpoint failed for {pair}: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.debug(f"Pricing API Response: {e.response.text}")
            except Exception as e:
                logger.warning(f"Error with pricing endpoint for {pair}: {str(e)}")
                
            logger.warning(f"No price data received from API for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {str(e)}", exc_info=True)
            return None
    
    def is_market_open(self) -> bool:
        """Check if it's within trading hours."""
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Check if current time is within trading hours
        if self.trading_hours_start <= current_hour < self.trading_hours_end:
            return True
        return False

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize the trading engine
        engine = OandaTradingEngine()
        
        # Get account balance
        balance = await engine.get_account_balance()
        print(f"Account Balance: {balance:.2f} {engine.account_currency}")
        
        # Get current price
        price = await engine.get_current_price()
        print(f"Current {engine.trading_pair} price: {price}")
        
        # Get historical data
        df = await engine.get_historical_data(count=10)
        if df is not None:
            print("\nLast 10 candles:")
            print(df[['open', 'high', 'low', 'close']].tail())
        
        # Example trade (commented out for safety)
        # trade = await engine.place_trade('buy')
        # print(f"Trade executed: {trade}")
        
        # Close all positions (commented out for safety)
        # await engine.close_all_positions()
    
    asyncio.run(main())
