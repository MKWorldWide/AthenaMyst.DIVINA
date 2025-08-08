import os
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import oandapyV20
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20.endpoints.accounts import AccountDetails, AccountSummary, AccountInstruments
from oandapyV20.endpoints.positions import OpenPositions, PositionDetails
from oandapyV20.endpoints.trades import OpenTrades, TradeDetails
from oandapyV20.endpoints.orders import OrderCreate, OrderCancel, OrderDetails
from dotenv import load_dotenv
try:
    from discord_webhook import DiscordWebhook
except ImportError:
    DiscordWebhook = None

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
    
    def __init__(self, config_file: str = '.env.oanda', discord_webhook_url: str = None):
        # Load configuration
        load_dotenv(config_file)
        
        # Initialize OANDA API client
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_type = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')
        
        # Set the correct environment URL based on account type
        environment = 'live' if self.account_type.lower() == 'live' else 'practice'
        self.client = API(access_token=self.api_key, environment=environment)
        
        logger.info(f"Initialized OANDA client for account: {self.account_id} ({environment})")
        
        # Trading parameters
        self.trading_pair = os.getenv('TRADING_PAIR', 'EUR_USD')
        self.account_currency = os.getenv('ACCOUNT_CURRENCY', 'USD')
        self.risk_percent = float(os.getenv('RISK_PERCENT', 10.0)) / 100.0  # Convert to decimal
        self.stop_loss_pips = float(os.getenv('STOP_LOSS_PIPS', 20))
        self.take_profit_pips = float(os.getenv('TAKE_PROFIT_PIPS', 40))
        
        # Trading hours (UTC)
        self.trading_hours_start = int(os.getenv('TRADING_HOURS_START', 0))
        self.trading_hours_end = int(os.getenv('TRADING_HOURS_END', 23))
        
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
        
        logger.info(f"Initialized OANDA Trading Engine for {self.trading_pair}")
    
    async def get_account_balance(self) -> float:
        """Get the current account balance."""
        try:
            # Get account details
            r = AccountDetails(accountID=self.account_id)
            response = self.client.request(r)
            
            # Update account metrics
            account = response.get('account', {})
            self.account_balance = float(account.get('balance', 0.0))
            self.unrealized_pnl = float(account.get('unrealizedPL', 0.0))
            self.margin_available = float(account.get('marginAvailable', 0.0))
            
            logger.debug(f"Account balance updated: {self.account_balance} {self.account_currency}")
            return self.account_balance
            
        except V20Error as e:
            logger.error(f"Error getting account balance: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API Response: {e.response.text}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in get_account_balance: {e}")
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
            
            # Convert to units
            units = (position_value / entry_price) * 100000  # Convert to standard lots (100,000 units per lot)
            
            # Round to nearest 1000 units (OANDA requirement)
            units = round(units / 1000) * 1000
            
            # Get maximum allowed position size from OANDA
            max_units = self.get_max_position_size()
            if max_units > 0:
                units = min(units, max_units)
                position_value = (units * entry_price) / 100000  # Recalculate position value based on capped units
            
            logger.info(f"Position size: {units:,} units (${position_value:,.2f} at {entry_price:.5f})")
            return units, position_value
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0, 0.0
            
    def get_max_position_size(self) -> float:
        """Get the maximum position size allowed by OANDA."""
        # OANDA typically has a maximum position size of 10,000,000 units for major pairs
        # This can be adjusted based on account type and instrument
        max_units = 10_000_000  # Default max units for major pairs
        
        # Adjust for minor pairs if needed
        if 'JPY' in self.trading_pair:
            max_units = 10_000_000  # Adjust for JPY pairs if needed
            
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
    
    async def get_historical_data(self, count: int = 100, granularity: str = "M5") -> Optional[pd.DataFrame]:
        """Get historical price data."""
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "MBA"  # Mid, Bid, Ask
            }
            
            # Create a list to store the candles
            candles = []
            
            # Use the InstrumentsCandlesFactory to get the data
            for r in InstrumentsCandlesFactory(instrument=self.trading_pair, params=params):
                self.client.request(r)
                for candle in r.response.get('candles', []):
                    candles.append({
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
            return df
            
        except V20Error as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    async def place_trade(self, direction: str, entry_price: float = None, reason: str = None) -> Optional[Dict]:
        """
        Place a trade with 10% position sizing and proper risk management.
        
        Args:
            direction: 'buy' or 'sell'
            entry_price: Optional entry price (if None, uses current market price)
            reason: Reason for the trade (for logging/notifications)
            
        Returns:
            Trade details or None if failed
        """
        try:
            # Get current price if not provided
            current_price = await self.get_current_price()
            if current_price is None:
                error_msg = "Could not get current price"
                logger.error(error_msg)
                if self.discord_webhook:
                    self.discord_webhook.send_error(error_msg)
                return None
                
            entry_price = entry_price or current_price
            
            # Calculate stop loss and take profit prices
            pip_multiplier = 0.0001  # For most currency pairs
            if 'JPY' in self.trading_pair:
                pip_multiplier = 0.01  # For JPY pairs
                
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
            
            # Send Discord notification
            if self.discord_webhook:
                self.discord_webhook.send_trade_opened(
                    pair=self.trading_pair,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    units=abs(units)
                )
            
            # Place the order
            r = OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)
            
            if 'orderFillTransaction' in response:
                logger.info(f"Successfully placed {direction.upper()} order")
                logger.debug(f"Order response: {response}")
                
                # Get the trade ID if available
                trade_id = None
                if 'tradeOpened' in response.get('orderFillTransaction', {}):
                    trade_id = response['orderFillTransaction']['tradeOpened'].get('tradeID')
                
                trade_details = {
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
                
                # Store active trade
                if trade_id:
                    self.active_trades[trade_id] = trade_details
                
                return trade_details
            else:
                error_msg = f"Failed to place order: {response}"
                logger.error(error_msg)
                if self.discord_webhook:
                    self.discord_webhook.send_error(error_msg)
                return None
            
        except V20Error as e:
            error_msg = f"Error placing {direction} order: {e}"
            logger.error(error_msg)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nAPI Response: {e.response.text}"
                logger.error(f"API Response: {e.response.text}")
            
            if self.discord_webhook:
                self.discord_webhook.send_error(error_msg)
                
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error in place_trade: {e}"
            logger.error(error_msg, exc_info=True)
            
            if self.discord_webhook:
                self.discord_webhook.send_error(error_msg)
                
            return None
            
    async def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on 10% account risk."""
        try:
            # Get account balance
            await self.get_account_balance()
            if self.account_balance <= 0:
                logger.error(f"Invalid account balance: {self.account_balance}")
                return 0.0
                
            # Calculate risk amount (10% of account balance)
            risk_amount = self.account_balance * self.risk_percent
            
            # Calculate position size
            if 'JPY' in self.trading_pair:
                pip_value = 0.01  # For JPY pairs
            else:
                pip_value = 0.0001  # For other currency pairs
                
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error(f"Invalid risk per unit: {risk_per_unit}")
                return 0.0
                
            position_value = (risk_amount / risk_per_unit) * entry_price
            
            # Convert to units (standard lot = 100,000 units)
            units = (position_value / entry_price) * 100000
            
            # Round to nearest 1000 units (OANDA requirement)
            units = round(units / 1000) * 1000
            
            logger.info(f"Position size: {units} units (${position_value:.2f} at {entry_price:.5f})")
            return units
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0
    
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
