"""
Trading loop implementation for the crypto trading bot.
Handles the main trading logic, position management, and monitoring.
"""
import os
import time
import ccxt
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
# Import all required functions from ccxt_engine
from ccxt_engine import (
    make_binanceus, 
    make_kraken, 
    discover_symbols,
    account_quote_balance,
    sized_amount,
    breakout_signal,
    place_market_order,
    start_exit_monitor,
    log_fill
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("LOG_FILE", "trading_bot.log"))
    ]
)
logger = logging.getLogger(__name__)

class TradingLoop:
    """Main trading loop implementation."""
    
    def __init__(self, exchange: ccxt.Exchange, quote_currency: str = None):
        """
        Initialize the trading loop.
        
        Args:
            exchange: Initialized CCXT exchange instance
            quote_currency: Default quote currency (e.g., 'USDT', 'USD')
        """
        self.exchange = exchange
        self.exchange_name = exchange.id
        self.quote_currency = quote_currency or ("USDT" if "binance" in self.exchange_name.lower() else "USD")
        self.cooldowns: Dict[str, float] = {}
        self.sell_cooldowns: Dict[str, float] = {}  # Track cooldowns after selling positions
        self.open_trades: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}  # Track open positions and their PnL
        self.last_market_reload = 0
        self.market_reload_interval = 1800  # 30 minutes in seconds
        
        # Balance management settings
        self.min_profit_threshold = float(os.getenv("MIN_PROFIT_THRESHOLD", "-0.5"))  # -0.5% minimum PnL to consider selling
        self.sell_cooldown_min = int(os.getenv("SELL_COOLDOWN_MIN", "15"))  # 15 minutes cooldown after selling
        self.max_sell_attempts = int(os.getenv("MAX_SELL_ATTEMPTS", "3"))  # Max sell attempts per position
        self.sell_attempts: Dict[str, int] = {}  # Track sell attempts per position
        
    def get_trade_symbols(self) -> List[str]:
        """
        Get the list of valid symbols to trade.
        
        Returns:
            List of valid trading pair symbols
        """
        # Check for explicitly configured symbols first
        env_var = f"{self.exchange_name.upper()}_SYMBOLS"
        logger.debug(f"Checking for symbols in environment variable: {env_var}")
        
        # Debug log all environment variables for troubleshooting
        logger.debug("=== Environment Variables ===")
        for key, value in os.environ.items():
            if key.startswith(self.exchange_name.upper()):
                logger.debug(f"{key}: {'*' * 8 + value[-4:] if 'KEY' in key or 'SECRET' in key else value}")
        logger.debug("============================")
        
        # Get the raw environment variable value
        env_value = os.getenv(env_var, '').strip()
        logger.debug(f"{env_var} raw value: '{env_var}='{env_value}'")
        
        # If the environment variable is not set or is empty, use auto-discovery
        if not env_value or env_value.startswith('#'):
            logger.info(f"{env_var} is empty or commented out, using auto-discovery")
        else:
            # Parse symbols, stripping whitespace and filtering out empty strings and comments
            symbols = []
            for s in env_value.split(','):
                s = s.strip()
                # Skip empty strings and comments
                if not s or s.startswith('#'):
                    logger.debug(f"Skipping empty or commented symbol: '{s}'")
                    continue
                # Check if the symbol is valid before adding it to the list
                if self.is_valid_symbol(s):
                    symbols.append(s)
                else:
                    logger.warning(f"Skipping invalid symbol: '{s}'")
            
            # If we have valid symbols, return them
            if symbols:
                logger.info(f"Using {len(symbols)} configured symbols: {symbols}")
                return symbols
            else:
                logger.warning(f"No valid symbols found in {env_var}, falling back to auto-discovery")
        
        # If we get here, either no symbols were configured or they were all invalid
        # So we'll discover symbols dynamically
        prefer_quotes = {"USDT", "USD"} if self.exchange_name.lower() == "binanceus" else {"USD"}
        max_price = float(os.getenv("MAX_PRICE", "5"))
        min_vol_usd = float(os.getenv("MIN_VOL_USD", "500000"))
        top_n = int(os.getenv("TOP_N_SYMBOLS", "25"))
        
        logger.info(f"Discovering top {top_n} symbols with: max_price=${max_price}, min_vol=${min_vol_usd}")
        
        try:
            # Try to discover symbols
            discovered_symbols = discover_symbols(
                self.exchange,
                prefer_quotes=prefer_quotes,
                max_price=max_price,
                min_vol_usd=min_vol_usd,
                top_n=top_n
            )
            
            # Filter out any invalid symbols (shouldn't be necessary, but just in case)
            valid_symbols = [s for s in discovered_symbols if self.is_valid_symbol(s)]
            
            if valid_symbols:
                logger.info(f"Discovered {len(valid_symbols)} valid symbols: {valid_symbols}")
                return valid_symbols
            else:
                logger.warning("No valid symbols discovered, using fallback symbols")
                
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}", exc_info=True)
            logger.warning("Symbol discovery failed, using fallback symbols")
        
        # If we get here, either discovery failed or returned no valid symbols
        # Return a safe default set of symbols
        fallback_symbols = ["BTC/USD", "ETH/USD"]
        logger.info(f"Using fallback symbols: {fallback_symbols}")
        return fallback_symbols
    
    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if a symbol is in cooldown.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if in cooldown, False otherwise
        """
        now = time.time()
        cooldown_end = self.cooldowns.get(symbol, 0)
        
        if now < cooldown_end:
            remaining = int(cooldown_end - now)
            logger.debug(f"{symbol} in cooldown for {remaining}s")
            return True
            
        return False
    
    def update_cooldown(self, symbol: str) -> None:
        """
        Update the cooldown for a symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        cooldown_min = int(os.getenv("COOLDOWN_MIN", "20"))
        self.cooldowns[symbol] = time.time() + (cooldown_min * 60)
        logger.debug(f"Set cooldown for {symbol} - {cooldown_min} minutes")
    
    def check_max_open_trades(self) -> bool:
        """
        Check if we've reached the maximum number of open trades.
        
        Returns:
            True if max open trades reached, False otherwise
        """
        max_open_trades = int(os.getenv("MAX_OPEN_TRADES", "6"))
        if len(self.open_trades) >= max_open_trades:
            logger.debug(f"Max open trades reached ({max_open_trades})")
            return True
        return False
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid for trading.
        
        Args:
            symbol: Trading pair symbol to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            logger.warning(f"Invalid symbol type: {symbol}")
            return False
            
        # Skip comments or empty strings
        if symbol.strip().startswith('#') or not symbol.strip():
            logger.warning(f"Skipping comment/empty symbol: '{symbol}'")
            return False
            
        # Check if symbol exists in exchange markets
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            logger.warning("Exchange markets not loaded")
            return False
            
        return symbol in self.exchange.markets
            
    def evaluate_positions(self) -> List[Dict[str, Any]]:
        """
        Evaluate all open positions and calculate their current PnL.
        
        Returns:
            List of position dictionaries sorted by PnL (least profitable first)
            with additional metadata for balance management
        """
        positions = []
        try:
            logger.info("Evaluating open positions for balance management...")
            
            # Get current ticker prices for all symbols
            tickers = self.exchange.fetch_tickers()
            
            # Get open orders to avoid selling positions with pending orders
            open_orders = self.exchange.fetch_open_orders()
            symbols_with_orders = {order['symbol'] for order in open_orders}
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            
            # Check all quote currencies (USD, USDT, etc.)
            for currency, amount in balance['total'].items():
                if amount <= 0:
                    continue
                    
                # Skip the quote currency itself
                if currency.upper() == self.quote_currency.upper():
                    continue
                    
                # Create symbol (e.g., 'BTC/USD')
                symbol = f"{currency}/{self.quote_currency}"
                
                # Skip if we don't have market data for this symbol
                if symbol not in self.exchange.markets:
                    logger.debug(f"Skipping {symbol}: Not in markets")
                    continue
                    
                # Skip if there are open orders for this symbol
                if symbol in symbols_with_orders:
                    logger.debug(f"Skipping {symbol}: Has open orders")
                    continue
                
                # Skip if in sell cooldown
                current_time = time.time()
                if symbol in self.sell_cooldowns and current_time < self.sell_cooldowns[symbol]:
                    remaining = int((self.sell_cooldowns[symbol] - current_time) / 60)  # in minutes
                    logger.debug(f"Skipping {symbol}: In sell cooldown for {remaining} more minutes")
                    continue
                    
                # Get current price with retry logic
                current_price = None
                for _ in range(3):  # Try up to 3 times
                    try:
                        if symbol in tickers:
                            current_price = tickers[symbol]['last']
                        else:
                            ticker = self.exchange.fetch_ticker(symbol)
                            current_price = ticker['last']
                        break
                    except Exception as e:
                        logger.warning(f"Error getting price for {symbol}: {e}")
                        time.sleep(1)  # Wait a bit before retry
                
                if current_price is None:
                    logger.error(f"Failed to get price for {symbol} after multiple attempts")
                    continue
                
                # Calculate PnL with more sophisticated logic
                try:
                    # Try to get actual entry price from open trades if available
                    entry_price = None
                    if symbol in self.open_trades:
                        trade = self.open_trades[symbol]
                        if 'entry_price' in trade:
                            entry_price = trade['entry_price']
                    
                    # Fallback to estimation if no entry price available
                    if entry_price is None:
                        # Assume 0.5% slippage on entry
                        entry_price = current_price * 0.995
                        
                        # If we have historical data, use it to improve the estimate
                        try:
                            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=24)  # Last 24 hours
                            if ohlcv and len(ohlcv) > 0:
                                # Use the lowest price in the last 24 hours as a conservative estimate
                                min_price = min(candle[3] for candle in ohlcv)  # Low price
                                entry_price = min(entry_price, min_price)
                        except Exception as e:
                            logger.debug(f"Couldn't fetch historical data for {symbol}: {e}")
                    
                    # Calculate PnL
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Calculate position value and PnL in quote currency
                    position_value = amount * current_price
                    pnl_value = (current_price - entry_price) * amount
                    
                    # Get position age if available
                    position_age = 0
                    if symbol in self.open_trades and 'entry_time' in self.open_trades[symbol]:
                        position_age = (time.time() - self.open_trades[symbol]['entry_time']) / 3600  # in hours
                    
                    positions.append({
                        'symbol': symbol,
                        'currency': currency,
                        'amount': amount,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'value': position_value,
                        'pnl_pct': pnl_pct,
                        'pnl_value': pnl_value,
                        'age_hours': position_age,
                        'last_updated': time.time()
                    })
                    
                    logger.debug(f"Evaluated {symbol}: {amount} @ {current_price} (Entry: {entry_price:.8f}, PnL: {pnl_pct:.2f}%)")
                    
                except Exception as e:
                    logger.error(f"Error evaluating position {symbol}: {e}", exc_info=True)
                    continue
            
            # Sort by PnL (least profitable first)
            positions.sort(key=lambda x: x['pnl_pct'])
            
        except Exception as e:
            logger.error(f"Error evaluating positions: {e}", exc_info=True)
            
        return positions
    
    def free_up_balance(self, required_amount: float) -> bool:
        """
        Try to free up balance by selling the least profitable positions.
        
        Args:
            required_amount: Amount of quote currency needed
            
        Returns:
            bool: True if enough balance was freed up, False otherwise
        """
        logger.info(f"Attempting to free up {required_amount} {self.quote_currency}")
        
        # Get current balance
        current_balance = account_quote_balance(self.exchange, self.quote_currency)
        if current_balance >= required_amount:
            logger.info(f"Sufficient balance available: {current_balance} {self.quote_currency}")
            return True
        
        logger.info(f"Current balance: {current_balance} {self.quote_currency}, Need additional: "
                  f"{required_amount - current_balance} {self.quote_currency}")
            
        # Get all positions sorted by PnL (least profitable first)
        positions = self.evaluate_positions()
        
        if not positions:
            logger.warning("No positions available to sell")
            return False
            
        logger.info(f"Evaluating {len(positions)} positions for potential sale")
        
        for position in positions:
            if current_balance >= required_amount:
                break
                
            symbol = position['symbol']
            position_value = position['value']
            pnl_pct = position['pnl_pct']
            
            # Check if position meets minimum profitability threshold
            if pnl_pct > self.min_profit_threshold:
                logger.info(f"Skipping {symbol}: PnL {pnl_pct:.2f}% is above threshold {self.min_profit_threshold}%")
                continue
                
            # Check sell attempts
            sell_attempts = self.sell_attempts.get(symbol, 0)
            if sell_attempts >= self.max_sell_attempts:
                logger.warning(f"Skipping {symbol}: Max sell attempts ({self.max_sell_attempts}) reached")
                continue
                
            logger.info(f"Selling {position['amount']:.8f} {position['currency']} ({symbol}) "
                      f"to free up ~{position_value:.2f} {self.quote_currency} (PnL: {pnl_pct:.2f}%)")
            
            try:
                # Place a market sell order with proper error handling
                order = place_market_order(
                    self.exchange,
                    symbol=symbol,
                    side='sell',
                    amount=position['amount']
                )
                
                if order and 'id' in order:
                    # Update sell attempts counter
                    self.sell_attempts[symbol] = sell_attempts + 1
                    
                    # Set cooldown for this symbol
                    self.sell_cooldowns[symbol] = time.time() + (self.sell_cooldown_min * 60)
                    
                    logger.info(f"Successfully placed sell order {order['id']} for {symbol}")
                    
                    # Update current balance estimate (we'll check actual balance after all sales)
                    current_balance += position_value
                    
                    # Update the position in our tracking
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
                    # Log the sale for record keeping
                    logger.info(f"Sold {position['amount']:.8f} {position['currency']} at {position['current_price']} "
                              f"(PnL: {pnl_pct:.2f}%, Value: {position_value:.2f} {self.quote_currency})")
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
                    
                else:
                    logger.error(f"Failed to place sell order for {symbol}")
                    self.sell_attempts[symbol] = sell_attempts + 1
                    
            except Exception as e:
                logger.error(f"Error selling {symbol}: {str(e)}", exc_info=True)
                self.sell_attempts[symbol] = sell_attempts + 1
                time.sleep(2)  # Wait a bit longer on error
        
        # Final balance check after all sales
        current_balance = account_quote_balance(self.exchange, self.quote_currency)
        if current_balance >= required_amount:
            logger.info(f"Successfully freed up balance. New balance: {current_balance} {self.quote_currency}")
            return True
            
        logger.warning(f"Could not free up enough balance. Current: {current_balance}, "
                      f"Required: {required_amount}, Short by: {required_amount - current_balance}")
        
        # Log remaining positions for debugging
        remaining_positions = [p for p in positions if p['symbol'] not in self.sell_cooldowns]
        if remaining_positions:
            logger.info("Remaining positions that could be sold (if they meet criteria):")
            for pos in sorted(remaining_positions, key=lambda x: x['pnl_pct']):
                logger.info(f"  {pos['symbol']}: {pos['amount']:.8f} @ {pos['current_price']:.8f} (PnL: {pos['pnl_pct']:.2f}%)")
        
        # Calculate PnL
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    
        # Calculate position value and PnL in quote currency
        position_value = amount * current_price
        pnl_value = (current_price - entry_price) * amount
        
        # Get position age if available
        position_age = 0
        if symbol in self.open_trades and 'entry_time' in self.open_trades[symbol]:
            position_age = (time.time() - self.open_trades[symbol]['entry_time']) / 3600  # in hours
        
        positions.append({
            'symbol': symbol,
            'currency': currency,
            'amount': amount,
            'entry_price': entry_price,
            'current_price': current_price,
            'value': position_value,
            'pnl_pct': pnl_pct,
            'pnl_value': pnl_value,
            'age_hours': position_age,
            'last_updated': time.time()
        })
    })
    
    logger.debug(f"Evaluated {symbol}: {amount} @ {current_price} (Entry: {entry_price:.8f}, PnL: {pnl_pct:.2f}%)")
    
except Exception as e:
    logger.error(f"Error evaluating position {symbol}: {e}", exc_info=True)
    continue

# Sort by PnL (least profitable first)
positions.sort(key=lambda x: x['pnl_pct'])

except Exception as e:
    logger.error(f"Error evaluating positions: {e}", exc_info=True)

return positions

def free_up_balance(self, required_amount: float) -> bool:
    """
    Try to free up balance by selling the least profitable positions.
    
    Args:
        required_amount: Amount of quote currency needed
        
    Returns:
        bool: True if enough balance was freed up, False otherwise
    """
    logger.info(f"Attempting to free up {required_amount} {self.quote_currency}")
    
    # Get current balance
    current_balance = account_quote_balance(self.exchange, self.quote_currency)
    if current_balance >= required_amount:
        logger.info(f"Sufficient balance available: {current_balance} {self.quote_currency}")
        return True
    
    logger.info(f"Current balance: {current_balance} {self.quote_currency}, Need additional: "
              f"{required_amount - current_balance} {self.quote_currency}")
        
    # Get all positions sorted by PnL (least profitable first)
    positions = self.evaluate_positions()
    
    if not positions:
        logger.warning("No positions available to sell")
        return False
        
    logger.info(f"Evaluating {len(positions)} positions for potential sale")
    
    for position in positions:
        if current_balance >= required_amount:
            break
            
        symbol = position['symbol']
        position_value = position['value']
        pnl_pct = position['pnl_pct']
        
        # Check if position meets minimum profitability threshold
        if pnl_pct > self.min_profit_threshold:
            logger.info(f"Skipping {symbol}: PnL {pnl_pct:.2f}% is above threshold {self.min_profit_threshold}%")
            continue
            
        # Check sell attempts
        sell_attempts = self.sell_attempts.get(symbol, 0)
        if sell_attempts >= self.max_sell_attempts:
            logger.warning(f"Skipping {symbol}: Max sell attempts ({self.max_sell_attempts}) reached")
            continue
            
        logger.info(f"Selling {position['amount']:.8f} {position['currency']} ({symbol}) "
                  f"to free up ~{position_value:.2f} {self.quote_currency} (PnL: {pnl_pct:.2f}%)")
        
        try:
            # Place a market sell order with proper error handling
            order = place_market_order(
                self.exchange,
                symbol=symbol,
                side='sell',
                amount=position['amount']
            )
            
            if order and 'id' in order:
                # Update sell attempts counter
                self.sell_attempts[symbol] = sell_attempts + 1
                
                # Set cooldown for this symbol
                self.sell_cooldowns[symbol] = time.time() + (self.sell_cooldown_min * 60)
                
                logger.info(f"Successfully placed sell order {order['id']} for {symbol}")
                
                # Update current balance estimate (we'll check actual balance after all sales)
                current_balance += position_value
                
                # Update the position in our tracking
                if symbol in self.positions:
                    del self.positions[symbol]
                    
                # Log the sale for record keeping
                logger.info(f"Sold {position['amount']:.8f} {position['currency']} at {position['current_price']} "
                          f"(PnL: {pnl_pct:.2f}%, Value: {position_value:.2f} {self.quote_currency})")
                
                # Small delay to avoid rate limits
                time.sleep(1)
                
            else:
                logger.error(f"Failed to place sell order for {symbol}")
                self.sell_attempts[symbol] = sell_attempts + 1
                
        except Exception as e:
            logger.error(f"Error selling {symbol}: {str(e)}", exc_info=True)
            self.sell_attempts[symbol] = sell_attempts + 1
            time.sleep(2)  # Wait a bit longer on error
    
    # Final balance check after all sales
    current_balance = account_quote_balance(self.exchange, self.quote_currency)
    if current_balance >= required_amount:
        logger.info(f"Successfully freed up balance. New balance: {current_balance} {self.quote_currency}")
        return True
        
    logger.warning(f"Could not free up enough balance. Current: {current_balance}, "
                  f"Required: {required_amount}, Short by: {required_amount - current_balance}")
    
    # Log remaining positions for debugging
    remaining_positions = [p for p in positions if p['symbol'] not in self.sell_cooldowns]
    if remaining_positions:
        logger.info("Remaining positions that could be sold (if they meet criteria):")
        for pos in sorted(remaining_positions, key=lambda x: x['pnl_pct']):
            logger.info(f"  {pos['symbol']}: {pos['amount']:.8f} @ {pos['current_price']} "
                      f"(PnL: {pos['pnl_pct']:.2f}%, Value: {pos['value']:.2f} {self.quote_currency})")
    
    return False

def process_symbol(self, symbol: str) -> None:
    """Process a single trading symbol.
    
    Args:
        symbol: Trading pair symbol
    """
    try:
        # Check if we've reached max open trades
        if self.check_max_open_trades():
            logger.debug(f"Max open trades reached, skipping {symbol}")
            return
            
        # Check if symbol is in cooldown
        if self.check_cooldown(symbol):
            return
            
        # Get OHLCV data
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
            if len(ohlcv) < 20:  # Need at least 20 candles
                logger.debug(f"Not enough data for {symbol}")
            logger.info(f"Max open trades reached ({max_open_trades}). Attempting to free up space...")
            # Try to close the least profitable position
            if not self.free_up_balance(0):  # Pass 0 to just close the least profitable position
                logger.warning(f"Max open trades reached ({max_open_trades}) and couldn't close any positions")
                return
            
            # Verify we have space now
            if len(self.open_trades) >= max_open_trades:
                logger.warning("Still at max open trades after attempting to free up space")
                return
            
            # Get OHLCV data with error handling
            try:
                timeframe = os.getenv("TIMEFRAME", "5m")
                limit = 100  # Get enough candles for indicators
                
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if len(ohlcv) < 50:  # Need enough data for indicators
                    logger.debug(f"Not enough data for {symbol} (got {len(ohlcv)} candles, need 50)")
                    return
                    
                # Get signal with better parameters for scalping
                signal, price = breakout_signal(ohlcv, lookback=14, buffer=0.0005)  # Reduced buffer for more signals
                
                # Add detailed logging for signal generation
                logger.debug(f"Signal check for {symbol}: {signal} at {price} (last close: {ohlcv[-1][4]})")
                
                if signal == "hold":
                    # Log why we're not taking the trade
                    last_high = max([c[2] for c in ohlcv[-15:-1]])  # High of last 14 candles
                    last_low = min([c[3] for c in ohlcv[-15:-1]])   # Low of last 14 candles
                    current_price = ohlcv[-1][4]  # Current close price
                    
                    logger.debug(f"No signal: current_price={current_price}, last_high={last_high}, last_low={last_low}")
                    logger.debug(f"Breakout levels: BUY > {last_high * 1.0005}, SELL < {last_low * 0.9995}")
                    return
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return
                    
                # Additional confirmation for scalping
                last_close = ohlcv[-1][4]  # Get the close of the last candle
                price = last_close  # Use close price for scalping
                
                # Get order book for better entry/exit points
                try:
                    orderbook = self.exchange.fetch_order_book(symbol, limit=5)
                    if signal == "buy":
                        # Use ask price for buys
                        price = orderbook['asks'][0][0] * 1.0005  # Slippage buffer
                    else:  # sell
                        # Use bid price for sells
                        price = orderbook['bids'][0][0] * 0.9995  # Slippage buffer
                except Exception as e:
                    logger.warning(f"Couldn't fetch order book for {symbol}, using last price: {e}")
                    price = last_close
                
                # Additional confirmation for scalping
                last_close = ohlcv[-1][4]  # Get the close of the last candle
                price = last_close  # Use close price for scalping
                
                # Get order book for better entry/exit points
                try:
                    orderbook = self.exchange.fetch_order_book(symbol, limit=5)
                    if signal == "buy":
                        # Use ask price for buys
                        price = orderbook['asks'][0][0] * 1.0005  # Slippage buffer
                    else:  # sell
                        # Use bid price for sells
                        price = orderbook['bids'][0][0] * 0.9995  # Slippage buffer
                except Exception as e:
                    logger.warning(f"Couldn't fetch order book for {symbol}, using last price: {e}")
                    price = last_close
                
                logger.info(f"{symbol} {signal.upper()} signal at {price}")
            
            except Exception as e:
                logger.error(f"Error getting OHLCV data for {symbol}: {e}")
                return
                
            # Calculate position size with aggressive but managed risk
            try:
                # Get current balance with detailed logging
                balance = account_quote_balance(self.exchange, self.quote_currency)
                logger.info(f"[process_symbol] {symbol} - Current {self.quote_currency} balance: {balance}")
                
                if balance <= 0:
                    logger.warning(f"[process_symbol] {symbol} - Insufficient {self.quote_currency} balance")
                    return
                
                # Dynamic position sizing based on performance
                risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% default
                logger.info(f"[process_symbol] {symbol} - Base risk per trade: {risk_per_trade*100}%")
                
                # Scale position size based on signal strength
                signal_strength = abs(ohlcv[-1][4] - ohlcv[-2][4]) / ohlcv[-2][4] if len(ohlcv) > 1 and ohlcv[-2][4] > 0 else 0
                logger.info(f"[process_symbol] {symbol} - Signal strength: {signal_strength*100:.2f}%")
                
                if signal_strength > 0.01:  # Strong signal
                    risk_per_trade = min(0.03, risk_per_trade * 1.5)  # Up to 3% for strong signals
                    logger.info(f"[process_symbol] {symbol} - Increased risk to {risk_per_trade*100}% for strong signal")
                
                notional = balance * risk_per_trade
                logger.info(f"[process_symbol] {symbol} - Calculated notional value: {notional} {self.quote_currency}")
                
                # Get initial amount with detailed logging
                logger.info(f"[process_symbol] {symbol} - Calling sized_amount with notional={notional}, price={price}")
                amount = sized_amount(self.exchange, symbol, notional, price)
                logger.info(f"[process_symbol] {symbol} - Initial sized amount: {amount}")
                
                # If not enough balance, try to free up funds
                if amount <= 0 or (amount * price) > balance:
                    logger.warning(f"[process_symbol] {symbol} - Insufficient balance. Current: {balance}, Needed: {notional}")
                    
                    # Calculate how much more we need
                    needed = max(0, notional - balance)
                    logger.info(f"[process_symbol] {symbol} - Attempting to free up {needed} {self.quote_currency}")
                    
                    # Try to free up the needed amount plus a buffer
                    if not self.free_up_balance(needed * 1.1):  # 10% buffer
                        logger.warning(f"[process_symbol] {symbol} - Could not free up enough balance")
                        return
                    
                    # Recalculate with new balance
                    balance = account_quote_balance(self.exchange, self.quote_currency)
                    logger.info(f"[process_symbol] {symbol} - New balance after freeing up: {balance} {self.quote_currency}")
                    
                    # Recalculate notional with potentially reduced balance
                    notional = min(balance * risk_per_trade, notional)
                    logger.info(f"[process_symbol] {symbol} - Recalculated notional: {notional} {self.quote_currency}")
                    
                    amount = sized_amount(self.exchange, symbol, notional, price)
                    logger.info(f"[process_symbol] {symbol} - Resized amount after balance check: {amount}")
                    
                    if amount <= 0:
                        logger.warning(f"[process_symbol] {symbol} - Still insufficient balance after attempting to free up funds")
                        return
                
                # Log position details before placing order
                position_value = amount * price
                logger.info(f"[process_symbol] {symbol} - Order details: {amount} @ ~{price} = {position_value} {self.quote_currency}")
                
                # Calculate precise entry with spread consideration
                ticker = self.exchange.fetch_ticker(symbol)
                if signal == 'buy':
                    # For buys, use ask price + small buffer to ensure fill
                    price = ticker['ask'] * 1.0005  # 0.05% above ask
                else:
                    # For sells, use bid price - small buffer to ensure fill
                    price = ticker['bid'] * 0.9995  # 0.05% below bid
                
                # Log the order details
                logger.info(f"Placing {order_type.upper()} {signal.upper()} order for {amount} {symbol} @ {price}")
                
                # Place the order
                order = place_market_order(
                    self.exchange,
                    symbol=symbol,
                    side=signal,
                    amount=amount
                )
                
                if order and 'id' in order:
                    logger.info(f"Successfully placed {signal} order: {order['id']}")
                    self.open_trades[symbol] = {
                        'order_id': order['id'],
                        'side': signal,
                        'amount': amount,
                        'price': price,
                        'timestamp': time.time(),
                        'status': 'open'
                    }
                    self.update_cooldown(symbol)
                    
                    # Start monitoring for exit conditions in a separate thread
                    try:
                        if signal == 'buy':  # Only monitor long positions for now
                            t = threading.Thread(
                                target=start_exit_monitor,
                                args=(
                                    self.exchange,
                                    symbol,
                                    signal,
                                    amount,
                                    price,
                                    float(os.getenv("TAKE_PROFIT_PCT", "0.0075")),  # 0.75% default
                                    float(os.getenv("STOP_LOSS_PCT", "0.0025")),    # 0.25% default
                                    True,  # Use trailing stop
                                    0.1    # 0.1% trailing stop
                                )
                            )
                            t.daemon = True
                            t.start()
                            logger.info(f"Started exit monitor thread for {symbol}")
                    except Exception as e:
                        logger.error(f"Error starting exit monitor for {symbol}: {e}", exc_info=True)
                else:
                    logger.error(f"Failed to place {signal} order for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error placing {signal} order for {symbol}: {e}", exc_info=True)
                
            # Clean up any closed trades for this symbol
            try:
                if symbol in self.open_trades:
                    # Check if the position is still open
                    positions = self.exchange.fetch_positions([symbol])
                    has_open = any(p.get('contracts', 0) > 0 for p in positions if p and isinstance(p, dict))
                    
                    if not has_open:
                        logger.info(f"Position closed: {symbol}")
                        del self.open_trades[symbol]
                        logger.info(f"Removed closed trade: {symbol}")
                        
            except Exception as e:
                logger.error(f"Error cleaning up position for {symbol}: {e}", exc_info=True)
                
            # Clean up any closed trades for this symbol
            try:
                if symbol in self.open_trades:
                    # Check if the position is still open
                    positions = self.exchange.fetch_positions([symbol])
                    has_open = any(p.get('contracts', 0) > 0 for p in positions if p and isinstance(p, dict))

                    if not has_open:
                        logger.info(f"Position closed: {symbol}")
                        del self.open_trades[symbol]
                        logger.info(f"Removed closed trade: {symbol}")
                        
            except Exception as e:
                logger.error(f"Error cleaning up position for {symbol}: {e}", exc_info=True)
                
            # Scale position size based on signal strength
            signal_strength = abs(ohlcv[-1][4] - ohlcv[-2][4]) / ohlcv[-2][4] if len(ohlcv) > 1 and ohlcv[-2][4] > 0 else 0
            logger.info(f"[process_symbol] {symbol} - Signal strength: {signal_strength*100:.2f}%")
            
            # Get current balance with detailed logging
            balance = account_quote_balance(self.exchange, self.quote_currency)
            logger.info(f"[process_symbol] {symbol} - Current {self.quote_currency} balance: {balance}")
            
            if balance <= 0:
                logger.warning(f"[process_symbol] {symbol} - Insufficient {self.quote_currency} balance")
                return
                
            # Dynamic position sizing based on performance
            risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% default
            logger.info(f"[process_symbol] {symbol} - Base risk per trade: {risk_per_trade*100}%")
            
            # Adjust risk based on signal strength
            if signal_strength > 0.01:  # Strong signal
                risk_per_trade = min(0.03, risk_per_trade * 1.5)  # Up to 3% for strong signals
                logger.info(f"[process_symbol] {symbol} - Increased risk to {risk_per_trade*100}% for strong signal")
                
            notional = balance * risk_per_trade
            logger.info(f"[process_symbol] {symbol} - Calculated notional value: {notional} {self.quote_currency}")
            
            try:
            # Get signal with better parameters for scalping
            signal, price = breakout_signal(ohlcv, lookback=14, buffer=0.0005)  # Reduced buffer for more signals
            
            # Add detailed logging for signal generation
            logger.debug(f"Signal check for {symbol}: {signal} at {price} (last close: {ohlcv[-1][4]})")
            
            if signal == "hold":
                # Log why we're not taking the trade
                last_high = max([c[2] for c in ohlcv[-15:-1]])  # High of last 14 candles
                last_low = min([c[3] for c in ohlcv[-15:-1]])   # Low of last 14 candles
                current_price = ohlcv[-1][4]  # Current close price
                
                logger.debug(f"No signal: current_price={current_price}, last_high={last_high}, last_low={last_low}")
                logger.debug(f"Breakout levels: BUY > {last_high * 1.0005}, SELL < {last_low * 0.9995}")
                return
                
            # Get order book for better entry/exit points
            try:
                orderbook = self.exchange.fetch_order_book(symbol, limit=5)
                if signal == "buy":
                    # Use ask price for buys
                    price = orderbook['asks'][0][0] * 1.0005  # Slippage buffer
                else:  # sell
                    # Use bid price for sells
                    price = orderbook['bids'][0][0] * 0.9995  # Slippage buffer
            except Exception as e:
                logger.warning(f"Couldn't fetch order book for {symbol}, using last price: {e}")
                price = ohlcv[-1][4]  # Fallback to last close price

            logger.info(f"{symbol} {signal.upper()} signal at {price}")
            
            # Calculate position size with aggressive but managed risk
            try:
                # Get current balance with detailed logging
                balance = account_quote_balance(self.exchange, self.quote_currency)
                logger.info(f"[process_symbol] {symbol} - Current {self.quote_currency} balance: {balance}")
                
                if balance <= 0:
                    logger.warning(f"[process_symbol] {symbol} - Insufficient {self.quote_currency} balance")
                    return
                
                # Dynamic position sizing based on performance
                risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% default
                logger.info(f"[process_symbol] {symbol} - Base risk per trade: {risk_per_trade*100}%")
                
                # Calculate position size based on risk and available balance
                try:
                    # Get current price for the symbol
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Calculate position size based on risk
                    risk_amount = balance * risk_per_trade
                    stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.0025"))  # 0.25% default
                    
                    # Calculate position size in base currency
                    position_size = (risk_amount / (current_price * stop_loss_pct)) * 0.99  # 1% buffer
                    
                    # Get market info for the symbol
                    market = self.exchange.market(symbol)
                    
                    # Apply precision and limits
                    if 'precision' in market and 'amount' in market['precision']:
                        position_size = float(self.exchange.amount_to_precision(symbol, position_size))
                    
                    # Ensure minimum order size
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    if min_amount and position_size < min_amount:
                        logger.warning(f"[process_symbol] {symbol} - Position size {position_size} below minimum {min_amount}")
                        return
                    
                    # Ensure sufficient balance
                    cost = position_size * current_price
                    if cost > balance * 0.99:  # 1% buffer for fees and slippage
                        logger.warning(f"[process_symbol] {symbol} - Insufficient balance for position size")
                        return
                    
                    logger.info(f"[process_symbol] {symbol} - Calculated position size: {position_size} (cost: {cost} {self.quote_currency})")
                    
                    # Place the order
                    try:
                        order = place_market_order(
                            self.exchange,
                            symbol=symbol,
                            side=signal.lower(),
                            amount=position_size,
                            price=price,
                            params={}
                        )
                        
                        if order and 'id' in order:
                            logger.info(f"Successfully placed {signal} order: {order['id']}")
                            self.open_trades[symbol] = {
                                'order_id': order['id'],
                                'side': signal,
                                'amount': position_size,
                                'price': price,
                                'timestamp': time.time(),
                                'status': 'open'
                            }
                            self.update_cooldown(symbol)
                        else:
                            logger.error(f"Failed to place {signal} order for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error placing {signal} order for {symbol}: {e}", exc_info=True)
                        
                except Exception as e:
                    logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
                    return
                    
            except Exception as e:
                logger.error(f"Error in trading logic for {symbol}: {e}", exc_info=True)
                return
                
            # Start exit monitor thread for this position
            try:
                if signal == 'buy':  # Only monitor long positions for now
                    t = threading.Thread(
                        target=start_exit_monitor,
                        args=(
                            self.exchange,
                            symbol,
                            signal,
                            amount,
                            price,
                            float(os.getenv("TAKE_PROFIT_PCT", "0.0075")),  # 0.75% default
                            float(os.getenv("STOP_LOSS_PCT", "0.0025")),    # 0.25% default
                            True,  # Use trailing stop
                            0.1    # 0.1% trailing stop
                        )
                    )
                    t.daemon = True
                    t.start()
                    logger.info(f"Started exit monitor thread for {symbol}")
            except Exception as e:
                logger.error(f"Error starting exit monitor for {symbol}: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error processing order for {symbol}: {e}", exc_info=True)
            
            # Clean up any closed trades for this symbol
            try:
                if symbol in self.open_trades:
                    # Check if the position is still open
                    positions = self.exchange.fetch_positions([symbol])
                    has_open = any(p.get('contracts', 0) > 0 for p in positions if p and isinstance(p, dict))

                    if not has_open:
                        logger.info(f"Position closed: {symbol}")
                        del self.open_trades[symbol]
                        logger.info(f"Removed closed trade: {symbol}")

            # Get initial amount with detailed logging
            logger.info(f"[process_symbol] {symbol} - Calling sized_amount with notional={notional}, price={price}")
            amount = sized_amount(self.exchange, symbol, notional, price)
            logger.info(f"[process_symbol] {symbol} - Initial sized amount: {amount}")
            
            # If not enough balance, try to free up funds
            if amount <= 0 or (amount * price) > balance:
                logger.warning(f"[process_symbol] {symbol} - Insufficient balance. Current: {balance}, Needed: {notional}")
                
                # Calculate how much more we need
                needed = max(0, notional - balance)
                logger.info(f"[process_symbol] {symbol} - Attempting to free up {needed} {self.quote_currency}")
                
                # Try to free up the needed amount plus a buffer
                if not self.free_up_balance(needed * 1.1):  # 10% buffer
                    logger.warning(f"[process_symbol] {symbol} - Could not free up enough balance")
                    return
                
                # Recalculate with new balance
                balance = account_quote_balance(self.exchange, self.quote_currency)
                logger.info(f"[process_symbol] {symbol} - New balance after freeing up: {balance} {self.quote_currency}")
                
                # Recalculate notional with potentially reduced balance
                notional = min(balance * risk_per_trade, notional)
                logger.info(f"[process_symbol] {symbol} - Recalculated notional: {notional} {self.quote_currency}")
                
                amount = sized_amount(self.exchange, symbol, notional, price)
                logger.info(f"[process_symbol] {symbol} - Resized amount after balance check: {amount}")

    if amount <= 0:
        logger.warning(f"[process_symbol] {symbol} - Still insufficient balance after attempting to free up funds")
        return

except Exception as e:
    logger.error(f"Error in trading logic for {symbol}: {e}", exc_info=True)
    return

except Exception as e:
    logger.error(f"Error processing signal for {symbol}: {e}", exc_info=True)
    return

def run(self) -> None:
    """Run the main trading loop."""
    logger.info(f"Starting trading loop for {self.exchange_name}")

    while True:
        try:
            # Reload markets periodically
            now = time.time()
            if now - self.last_market_reload > self.market_reload_interval:
                try:
                    self.exchange.load_markets(True)
                    self.last_market_reload = now
                    logger.info("Reloaded markets")
                except Exception as e:
                    logger.error(f"Error reloading markets: {e}")
            
            # Get list of symbols to trade
            symbols = self.get_trade_symbols()
            if not symbols:
                logger.warning("No valid trading symbols found")
                time.sleep(60)  # Wait a minute before retrying
                continue
            
            # Process each symbol with error handling
            for symbol in symbols:
                try:
                    self.process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            
            # Clean up closed trades
            self.cleanup_closed_trades()
            
            # Sleep before next iteration
            sleep_time = int(os.getenv("LOOP_INTERVAL", "10"))  # seconds
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(30)  # Avoid tight loop on error
                    # Check if the position is still open
                    positions = self.exchange.fetch_positions([symbol])
                    has_open = any(p.get('contracts', 0) > 0 for p in positions if p and isinstance(p, dict))
                    
                    if not has_open:
                        logger.info(f"Position closed: {symbol}")
                        del self.open_trades[symbol]
                        logger.info(f"Removed closed trade: {symbol}")
                        
            except Exception as e:
                logger.error(f"Error cleaning up position for {symbol}: {e}", exc_info=True)
    
    def run(self) -> None:
        """Run the main trading loop."""
        logger.info(f"Starting trading loop for {self.exchange_name}")
        
        while True:
            try:
                # Reload markets periodically
                now = time.time()
                if now - self.last_market_reload > self.market_reload_interval:
                    try:
                        self.exchange.load_markets(True)
                        self.last_market_reload = now
                        logger.info("Reloaded markets")
                    except Exception as e:
                        logger.error(f"Error reloading markets: {e}")
                
                # Get list of symbols to trade
                symbols = self.get_trade_symbols()
                if not symbols:
                    logger.warning("No valid trading symbols found")
                    time.sleep(60)  # Wait a minute before retrying
                    continue
                
                # Process each symbol with error handling
                for symbol in symbols:
                    try:
                        self.process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
                # Clean up closed trades
                self.cleanup_closed_trades()
                
                # Sleep before next iteration
                sleep_time = int(os.getenv("LOOP_INTERVAL", "10"))  # seconds
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(30)  # Avoid tight loop on error


def run_kraken(logger=None) -> None:
    """Run the trading loop for Kraken.
    
    Args:
        logger: Optional logger instance to use. If not provided, a default will be used.
    """
    if logger is None:
        logger = logging.getLogger("trading_loop.kraken")
        
    try:
        exchange = make_kraken()
        loop = TradingLoop(exchange, "USD")
        logger.info("Starting Kraken trading loop")
        loop.run()
    except Exception as e:
        logger.error(f"Fatal error in Kraken trading loop: {e}", exc_info=True)
        raise


def run_binanceus(logger=None) -> None:
    """Run the trading loop for Binance.US.
    
    Args:
        logger: Optional logger instance to use. If not provided, a default will be used.
    """
    if logger is None:
        logger = logging.getLogger("trading_loop.binanceus")
        
    try:
        exchange = make_binanceus()
        loop = TradingLoop(exchange, "USDT")
        logger.info("Starting Binance.US trading loop")
        loop.run()
    except Exception as e:
        logger.error(f"Fatal error in Binance.US trading loop: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--exchange", choices=["kraken", "binanceus"], required=True,
                       help="Exchange to trade on")
    
    args = parser.parse_args()
    
    if args.exchange == "kraken":
        run_kraken()
    elif args.exchange == "binanceus":
        run_binanceus()
