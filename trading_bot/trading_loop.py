"""
Trading loop implementation for the crypto trading bot.
Handles the main trading logic, position management, and monitoring.
"""
import os
import time
import ccxt
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

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

@dataclass
class TradeSignal:
    """Data class to represent a trading signal."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    confidence: float
    indicators: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

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
        self.sell_cooldowns: Dict[str, float] = {}
        self.open_trades: Dict[str, Dict] = {}
        self.config = self._load_config()
        
        # Initialize exchange settings
        self._init_exchange_settings()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.0025')),
            'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.0075')),
            'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '6')),
            'min_profit_threshold': float(os.getenv('MIN_PROFIT_THRESHOLD', '-0.5')),
            'sell_cooldown_min': int(os.getenv('SELL_COOLDOWN_MIN', '15')),
            'max_sell_attempts': int(os.getenv('MAX_SELL_ATTEMPTS', '3')),
            'ohlcv_limit': int(os.getenv('OHLCV_LIMIT', '100')),
            'timeframe': os.getenv('TIMEFRAME', '1m'),
        }
    
    def _init_exchange_settings(self):
        """Initialize exchange-specific settings."""
        # Set exchange options
        self.exchange.enableRateLimit = True
        self.exchange.options['adjustForTimeDifference'] = True
        
        # Load markets if not already loaded
        if not hasattr(self.exchange, 'markets'):
            self.exchange.load_markets()
    
    def check_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is in cooldown period."""
        current_time = time.time()
        if symbol in self.cooldowns and current_time < self.cooldowns[symbol]:
            remaining = self.cooldowns[symbol] - current_time
            logger.debug(f"{symbol} in cooldown for {remaining:.1f}s")
            return True
        return False
    
    def check_max_open_trades(self) -> bool:
        """Check if maximum number of open trades has been reached."""
        active_trades = len([t for t in self.open_trades.values() if t.get('status') == 'open'])
        if active_trades >= self.config['max_open_trades']:
            logger.debug(f"Max open trades reached ({active_trades}/{self.config['max_open_trades']})")
            return True
        return False
    
    def process_symbol(self, symbol: str) -> None:
        """
        Process a single trading symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USD')
        """
        try:
            # Check if we can take new trades
            if self.check_max_open_trades():
                logger.debug(f"Max open trades reached, skipping {symbol}")
                return
                
            if self.check_cooldown(symbol):
                return
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=self.config['timeframe'], 
                limit=self.config['ohlcv_limit']
            )
            
            if len(ohlcv) < 20:  # Minimum data points needed
                logger.debug(f"Not enough data for {symbol}")
                return
            
            # Generate trading signal
            signal, price = breakout_signal(ohlcv, lookback=14, buffer=0.0005)
            logger.info(f"{symbol} {signal.upper()} signal at {price}")
            
            # Process the signal
            if signal == 'buy':
                self._process_buy_signal(symbol, price)
            elif signal == 'sell':
                self._process_sell_signal(symbol, price)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    def _process_buy_signal(self, symbol: str, price: float) -> None:
        """Process a buy signal."""
        try:
            # Calculate position size
            balance = account_quote_balance(self.exchange, self.quote_currency)
            risk_amount = balance * self.config['risk_per_trade']
            stop_loss = price * (1 - self.config['stop_loss_pct'])
            
            # Calculate position size with 1% buffer for slippage/fees
            position_size = (risk_amount / (price - stop_loss)) * 0.99
            
            if position_size <= 0:
                logger.warning(f"Invalid position size {position_size} for {symbol}")
                return
                
            # Place buy order
            logger.info(f"Placing BUY order for {symbol}: {position_size:.8f} @ {price:.8f}")
            order = place_market_order(
                self.exchange, 
                symbol, 
                'buy', 
                position_size, 
                price
            )
            
            if order and 'id' in order:
                # Start exit monitoring for this position
                start_exit_monitor(
                    self.exchange, 
                    symbol, 
                    order['id'], 
                    price, 
                    self.config['stop_loss_pct'], 
                    self.config['take_profit_pct']
                )
                
                # Record the trade
                self.open_trades[symbol] = {
                    'entry_price': price,
                    'amount': position_size,
                    'entry_time': time.time(),
                    'status': 'open',
                    'order_id': order['id']
                }
                
                logger.info(f"Successfully opened position in {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing BUY signal for {symbol}: {e}", exc_info=True)
    
    def _process_sell_signal(self, symbol: str, price: float) -> None:
        """Process a sell signal."""
        try:
            if symbol not in self.open_trades:
                logger.debug(f"No open position found for {symbol}")
                return
                
            position = self.open_trades[symbol]
            if position['status'] != 'open':
                logger.debug(f"Position for {symbol} is not open")
                return
                
            # Place sell order
            logger.info(f"Placing SELL order for {symbol}: {position['amount']:.8f} @ {price:.8f}")
            order = place_market_order(
                self.exchange, 
                symbol, 
                'sell', 
                position['amount'], 
                price
            )
            
            if order and 'id' in order:
                # Update trade status
                position['status'] = 'closed'
                position['exit_price'] = price
                position['exit_time'] = time.time()
                position['pnl_pct'] = ((price - position['entry_price']) / position['entry_price']) * 100
                
                logger.info(f"Successfully closed position in {symbol}: "
                          f"P&L: {position['pnl_pct']:.2f}%")
                
                # Set cooldown for this symbol
                self.cooldowns[symbol] = time.time() + (self.config['sell_cooldown_min'] * 60)
                
        except Exception as e:
            logger.error(f"Error processing SELL signal for {symbol}: {e}", exc_info=True)
    
    def run(self, symbols: List[str] = None) -> None:
        """
        Run the main trading loop.
        
        Args:
            symbols: List of symbols to trade. If None, will discover symbols.
        """
        logger.info("Starting trading loop...")
        
        if not symbols:
            symbols = discover_symbols(self.exchange, self.quote_currency)
            logger.info(f"Discovered {len(symbols)} symbols to trade")
        
        try:
            while True:
                try:
                    for symbol in symbols:
                        try:
                            self.process_symbol(symbol)
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                            time.sleep(5)
                    
                    # Clean up closed trades periodically
                    self._cleanup_closed_trades()
                    time.sleep(5)  # Small delay between full symbol cycles
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    time.sleep(30)  # Longer delay on critical errors
                    
        except KeyboardInterrupt:
            logger.info("Trading loop stopped by user")
        except Exception as e:
            logger.critical(f"Fatal error in trading loop: {e}", exc_info=True)
        finally:
            # Clean up resources
            self.exchange.close()
    
    def _cleanup_closed_trades(self) -> None:
        """Remove closed trades from the open_trades dictionary."""
        closed_trades = [
            symbol for symbol, trade in self.open_trades.items() 
            if trade.get('status') == 'closed'
        ]
        
        for symbol in closed_trades:
            del self.open_trades[symbol]
            
        if closed_trades:
            logger.debug(f"Cleaned up {len(closed_trades)} closed trades")

def run_kraken(quote_currency: str = 'USD') -> None:
    """Run the trading loop for Kraken."""
    try:
        exchange = make_kraken()
        bot = TradingLoop(exchange, quote_currency)
        bot.run()
    except Exception as e:
        logger.critical(f"Failed to start Kraken trading bot: {e}", exc_info=True)
        raise

def run_binanceus(quote_currency: str = 'USDT') -> None:
    """Run the trading loop for Binance.US."""
    try:
        exchange = make_binanceus()
        bot = TradingLoop(exchange, quote_currency)
        bot.run()
    except Exception as e:
        logger.critical(f"Failed to start Binance.US trading bot: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the trading bot')
    parser.add_argument('--exchange', type=str, default='kraken',
                      choices=['kraken', 'binanceus'],
                      help='Exchange to trade on')
    parser.add_argument('--quote', type=str, default=None,
                      help='Quote currency (default: USD for Kraken, USDT for Binance.US)')
    
    args = parser.parse_args()
    
    if args.exchange == 'kraken':
        run_kraken(args.quote or 'USD')
    elif args.exchange == 'binanceus':
        run_binanceus(args.quote or 'USDT')
