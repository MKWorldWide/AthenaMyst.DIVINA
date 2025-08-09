"""
Trading loop implementation for the crypto trading bot.
Handles the main trading logic, position management, and monitoring.
"""
import os
import time
import ccxt
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
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
        self.open_trades: Dict[str, Dict] = {}
        self.last_market_reload = 0
        self.market_reload_interval = 1800  # 30 minutes in seconds
        
    def get_trade_symbols(self) -> List[str]:
        """
        Get the list of symbols to trade.
        
        Returns:
            List of trading pair symbols
        """
        # Check for explicitly configured symbols
        env_var = f"{self.exchange_name.upper()}_SYMBOLS"
        if env_var in os.environ and os.environ[env_var]:
            return [s.strip() for s in os.environ[env_var].split(",") if s.strip()]
        
        # Otherwise discover symbols dynamically
        prefer_quotes = {"USDT", "USD"} if self.exchange_name.lower() == "binanceus" else {"USD"}
        max_price = float(os.getenv("MAX_PRICE", "5"))
        min_vol_usd = float(os.getenv("MIN_VOL_USD", "500000"))
        top_n = int(os.getenv("TOP_N_SYMBOLS", "25"))
        
        try:
            symbols = discover_symbols(
                self.exchange,
                prefer_quotes=prefer_quotes,
                max_price=max_price,
                min_vol_usd=min_vol_usd,
                top_n=top_n
            )
            logger.info(f"Discovered {len(symbols)} symbols to trade")
            return symbols
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []
    
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
    
    def process_symbol(self, symbol: str) -> None:
        """
        Process a single trading symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Skip if in cooldown or max trades reached
            if self.check_cooldown(symbol) or self.check_max_open_trades():
                return
            
            # Get OHLCV data
            timeframe = os.getenv("TIMEFRAME", "5m")
            limit = 100  # Get enough candles for indicators
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if len(ohlcv) < 50:  # Need enough data for indicators
                logger.debug(f"Not enough data for {symbol}")
                return
            
            # Get signal
            signal, price = breakout_signal(ohlcv)
            if signal == "hold":
                return
            
            logger.info(f"{symbol} {signal.upper()} signal at {price}")
            
            # Calculate position size
            balance = account_quote_balance(self.exchange, self.quote_currency)
            risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.0125"))  # 1.25%
            notional = balance * risk_per_trade
            
            amount = sized_amount(self.exchange, symbol, notional, price)
            if amount <= 0:
                logger.warning(f"Invalid amount for {symbol}: {amount}")
                self.update_cooldown(symbol)
                return
            
            # Place order
            order = place_market_order(self.exchange, symbol, signal, amount)
            if not order:
                logger.error(f"Failed to place {signal} order for {symbol}")
                self.update_cooldown(symbol)
                return
            
            # Start exit monitor in background
            tp_pct = float(os.getenv("TP_PCT", "0.004"))  # 0.4%
            sl_pct = float(os.getenv("SL_PCT", "0.003"))   # 0.3%
            
            start_exit_monitor(
                self.exchange,
                symbol=symbol,
                side=signal,
                amount=amount,
                entry=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct
            )
            
            # Update state
            self.open_trades[symbol] = {
                'side': signal,
                'amount': amount,
                'entry': price,
                'timestamp': time.time()
            }
            
            self.update_cooldown(symbol)
            logger.info(f"Opened {signal} position: {amount} {symbol} @ {price}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            self.update_cooldown(symbol)
    
    def cleanup_closed_trades(self) -> None:
        """Remove any closed trades from the open_trades dict."""
        closed = []
        for symbol in list(self.open_trades.keys()):
            try:
                positions = self.exchange.fetch_positions([symbol])
                has_open = any(p['contracts'] > 0 for p in positions if p and isinstance(p, dict))
                if not has_open:
                    closed.append(symbol)
            except Exception as e:
                logger.error(f"Error checking position for {symbol}: {e}")
        
        for symbol in closed:
            if symbol in self.open_trades:
                del self.open_trades[symbol]
                logger.info(f"Removed closed trade: {symbol}")
    
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
                
                # Get symbols to trade
                symbols = self.get_trade_symbols()
                if not symbols:
                    logger.warning("No symbols to trade")
                    time.sleep(60)
                    continue
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        self.process_symbol(symbol)
                        # Small delay between symbols to avoid rate limits
                        time.sleep(1)
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


def run_kraken() -> None:
    """Run the trading loop for Kraken."""
    try:
        exchange = make_kraken()
        loop = TradingLoop(exchange, "USD")
        loop.run()
    except Exception as e:
        logger.error(f"Fatal error in Kraken trading loop: {e}", exc_info=True)
        raise


def run_binanceus() -> None:
    """Run the trading loop for Binance.US."""
    try:
        exchange = make_binanceus()
        loop = TradingLoop(exchange, "USDT")
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
