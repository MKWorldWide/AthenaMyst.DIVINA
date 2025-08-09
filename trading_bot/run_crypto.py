#!/usr/bin/env python3
"""
24/7 Crypto Trading Engine

This script implements a multi-exchange crypto trading bot that:
1. Discovers trading opportunities on Binance.US and Kraken
2. Executes trades based on breakout signals
3. Manages risk with position sizing and stop-losses
4. Runs 24/7 with automatic error recovery
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ccxt
from dotenv import load_dotenv

from ccxt_engine import (
    make_binanceus,
    make_kraken,
    discover_symbols,
    account_quote_balance,
    position_size,
    breakout_signal,
    place_market_order
)
from exit_monitor import TradeState, ExitMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoEngine')

# Load environment variables
load_dotenv()

# Configuration from environment variables
CONFIG = {
    'max_price': float(os.getenv('MAX_PRICE', '5.0')),  # Max price per coin
    'min_volume': float(os.getenv('MIN_VOL_USD', '500000')),  # Min 24h volume in USD
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.01')),  # 1% risk per trade
    'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '3')),  # Max concurrent trades
    'timeframe': os.getenv('TIMEFRAME', '5m'),  # Candle timeframe
    'tp_pct': float(os.getenv('TP_PCT', '0.004')),  # 0.4% take profit
    'sl_pct': float(os.getenv('SL_PCT', '0.003')),  # 0.3% stop loss
    'cooldown_min': int(os.getenv('COOLDOWN_MIN', '20')),  # Cooldown between trades on same pair (minutes)
}

class CryptoTradingEngine:
    """Main trading engine for 24/7 crypto trading."""
    
    def __init__(self):
        """Initialize the trading engine with exchanges and state."""
        self.exchanges = {}
        self.active_trades = {}
        self.cooldowns = {}
        self.threads = []
        self.running = False
        
        # Initialize exchanges
        self._init_exchanges()
        
    def _init_exchanges(self):
        """Initialize exchange connections."""
        try:
            self.exchanges['binanceus'] = make_binanceus()
            logger.info("Connected to Binance.US")
        except Exception as e:
            logger.error(f"Failed to connect to Binance.US: {e}")
            
        try:
            self.exchanges['kraken'] = make_kraken()
            logger.info("Connected to Kraken")
        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
    
    def discover_opportunities(self):
        """Discover trading opportunities across all exchanges."""
        opportunities = []
        
        for ex_name, exchange in self.exchanges.items():
            try:
                # Get available symbols that match our criteria
                symbols = discover_symbols(
                    exchange,
                    max_price=CONFIG['max_price'],
                    min_vol_usd=CONFIG['min_volume']
                )
                
                logger.info(f"Discovered {len(symbols)} symbols on {ex_name}: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
                
                # Check each symbol for trading signals
                for symbol in symbols:
                    # Skip if we're already in a trade for this symbol
                    if symbol in self.active_trades:
                        continue
                        
                    # Check cooldown
                    if self._is_in_cooldown(symbol):
                        continue
                    
                    # Get OHLCV data
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=50)
                    if len(ohlcv) < 20:  # Need at least 20 candles for analysis
                        continue
                    
                    # Get signal
                    signal, price = breakout_signal(ohlcv)
                    if signal == 'hold':
                        continue
                    
                    # Calculate position size
                    quote_currency = symbol.split('/')[1]
                    balance = account_quote_balance(exchange, quote_currency)
                    risk_amount = balance * CONFIG['risk_per_trade']
                    amount = position_size(exchange, symbol, risk_amount, price)
                    
                    if amount <= 0:
                        continue
                    
                    # Add to opportunities
                    opportunities.append({
                        'exchange': ex_name,
                        'symbol': symbol,
                        'side': signal,
                        'price': price,
                        'amount': amount,
                        'take_profit': price * (1 + CONFIG['tp_pct']) if signal == 'buy' else price * (1 - CONFIG['tp_pct']),
                        'stop_loss': price * (1 - CONFIG['sl_pct']) if signal == 'buy' else price * (1 + CONFIG['sl_pct']),
                    })
                    
            except Exception as e:
                logger.error(f"Error discovering opportunities on {ex_name}: {e}")
        
        return opportunities
    
    def execute_trade(self, trade: dict):
        """Execute a trade and start monitoring for exit."""
        exchange = self.exchanges[trade['exchange']]
        symbol = trade['symbol']
        
        try:
            # Place the market order
            logger.info(f"Placing {trade['side']} order for {symbol} @ ~{trade['price']:.8f} "
                      f"(TP: {trade['take_profit']:.8f}, SL: {trade['stop_loss']:.8f})")
            
            order = place_market_order(
                exchange=exchange,
                symbol=symbol,
                side=trade['side'],
                amount=trade['amount']
            )
            
            if not order:
                raise Exception("Failed to place order")
            
            # Create trade state
            trade_state = TradeState(
                symbol=symbol,
                side=trade['side'],
                entry_price=float(order['price'] or trade['price']),
                amount=float(order['filled'] or trade['amount']),
                take_profit=trade['take_profit'],
                stop_loss=trade['stop_loss'],
                timestamp=time.time()
            )
            
            # Start exit monitor in a new thread
            monitor = ExitMonitor(exchange, trade_state)
            monitor_thread = threading.Thread(
                target=monitor.run,
                daemon=True
            )
            
            # Store active trade and thread
            self.active_trades[symbol] = {
                'monitor': monitor,
                'thread': monitor_thread,
                'exchange': trade['exchange']
            }
            
            # Start monitoring
            monitor_thread.start()
            self.threads.append(monitor_thread)
            
            # Add cooldown for this symbol
            self._add_cooldown(symbol, CONFIG['cooldown_min'] * 60)
            
            logger.info(f"Successfully opened {trade['side']} position on {symbol} "
                      f"@ {trade_state.entry_price:.8f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            # Add cooldown on error to prevent rapid retries
            self._add_cooldown(symbol, 300)  # 5 minute cooldown on error
            return False
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is in cooldown."""
        if symbol not in self.cooldowns:
            return False
            
        remaining = self.cooldowns[symbol] - time.time()
        if remaining > 0:
            logger.debug(f"{symbol} in cooldown for {int(remaining)}s")
            return True
            
        # Clean up expired cooldowns
        if remaining <= 0:
            del self.cooldowns[symbol]
            
        return False
    
    def _add_cooldown(self, symbol: str, duration_seconds: int):
        """Add a cooldown period for a symbol."""
        self.cooldowns[symbol] = time.time() + duration_seconds
    
    def cleanup(self):
        """Clean up resources and close open positions."""
        self.running = False
        
        # Stop all monitor threads
        for symbol, trade in list(self.active_trades.items()):
            try:
                trade['monitor'].stop()
                trade['thread'].join(timeout=5)
            except Exception as e:
                logger.error(f"Error cleaning up {symbol}: {e}")
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            try:
                if hasattr(exchange, 'close'):
                    exchange.close()
            except Exception as e:
                logger.error(f"Error closing exchange: {e}")
    
    def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting crypto trading engine")
        
        try:
            while self.running:
                try:
                    # Clean up finished trades
                    self.active_trades = {
                        k: v for k, v in self.active_trades.items()
                        if v['thread'].is_alive()
                    }
                    
                    # Skip if we're at max open trades
                    if len(self.active_trades) >= CONFIG['max_open_trades']:
                        time.sleep(5)
                        continue
                    
                    # Discover trading opportunities
                    opportunities = self.discover_opportunities()
                    
                    # Execute trades if we have opportunities and capacity
                    for opp in opportunities[:CONFIG['max_open_trades'] - len(self.active_trades)]:
                        if not self.running:
                            break
                            
                        if self.execute_trade(opp):
                            # Small delay between trade executions
                            time.sleep(1)
                    
                    # Throttle the main loop
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(30)  # Wait before retrying on error
                    
        finally:
            self.cleanup()
            logger.info("Trading engine stopped")

if __name__ == "__main__":
    # Create and run the trading engine
    engine = CryptoTradingEngine()
    
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        engine.cleanup()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        engine.cleanup()
