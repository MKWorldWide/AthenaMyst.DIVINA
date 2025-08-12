import os, time, math, threading, csv, datetime
from typing import Dict, Any, List, Tuple, Optional, Set, Union, Callable
from dotenv import load_dotenv
import ccxt
import logging

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

load_dotenv()

# Stablecoins to exclude from trading
STABLES = {"USDT", "USDC", "USD", "FDUSD", "DAI", "TUSD", "BUSD"}

def make_binanceus() -> ccxt.Exchange:
    """
    Initialize and return a Binance.US exchange instance with proper configuration.
    Handles recvWindow, time sync, and market reload.
    """
    ex = ccxt.binanceus({
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
            "defaultType": "spot",
            "recvWindow": int(os.getenv("BINANCE_RECVWINDOW", "15000")),
        },
        # Add timestamp for requests to prevent replay attacks
        "timeDifference": None,  # Let CCXT handle time sync
    })
    # Force reload markets to ensure we have latest precision info
    ex.load_markets(True)
    return ex

def make_kraken() -> ccxt.Exchange:
    """
    Initialize and return a Kraken exchange instance with proper configuration.
    Handles time sync and market reload.
    """
    # Debug log environment variables
    logger.debug("=== Kraken Environment Variables ===")
    logger.debug(f"KRAKEN_API_KEY: {'*' * 8 + os.getenv('KRAKEN_API_KEY', '')[-4:] if os.getenv('KRAKEN_API_KEY') else 'Not set'}")
    logger.debug(f"KRAKEN_API_SECRET: {'*' * 8 + os.getenv('KRAKEN_API_SECRET', '')[-4:] if os.getenv('KRAKEN_API_SECRET') else 'Not set'}")
    logger.debug(f"KRAKEN_SYMBOLS: {os.getenv('KRAKEN_SYMBOLS', 'Not set')}")
    logger.debug("==================================")
    
    try:
        ex = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY"),
            "secret": os.getenv("KRAKEN_API_SECRET"),
            "enableRateLimit": True,
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 60000,  # Kraken default is 60s
            },
        })
    except Exception as e:
        logger.error(f"Failed to initialize Kraken exchange: {e}", exc_info=True)
        raise
    # Force reload markets to ensure we have latest precision info
    ex.load_markets(True)
    
    # Start a background thread to reload markets every 30 minutes
    def reload_markets():
        while True:
            time.sleep(1800)  # 30 minutes
            try:
                ex.load_markets(True)
            except Exception as e:
                print(f"Error reloading markets: {e}")
    
    threading.Thread(target=reload_markets, daemon=True).start()
    return ex

def discover_symbols(
    ex: ccxt.Exchange, 
    prefer_quotes: Set[str] = None,
    max_price: float = None,
    min_vol_usd: float = None,
    top_n: int = 25
) -> List[str]:
    """
    Discover top trading symbols by volume that match criteria.
    
    Args:
        ex: The exchange instance
        prefer_quotes: Preferred quote currencies (e.g., {"USDT", "USD"})
        max_price: Maximum price in quote currency (default from env)
        min_vol_usd: Minimum 24h volume in USD (default from env)
        top_n: Maximum number of symbols to return
        
    Returns:
        List of symbol strings sorted by volume (highest first)
    """
    logger.info(f"Starting symbol discovery for {ex.id}...")
    
    if prefer_quotes is None:
        prefer_quotes = {"USDT", "USD"}
    if max_price is None:
        max_price = float(os.getenv("MAX_PRICE", "5"))
    if min_vol_usd is None:
        min_vol_usd = float(os.getenv("MIN_VOL_USD", "500000"))
        
    logger.info(f"Discovery params - Prefer quotes: {prefer_quotes}, Max price: ${max_price}, Min volume: ${min_vol_usd}, Top N: {top_n}")
    
    try:
        tickers = ex.fetch_tickers()
        logger.info(f"Fetched {len(tickers)} tickers from {ex.id}")
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return []
        
    picks = []
    
    for symbol, t in tickers.items():
        try:
            # Skip if market data is incomplete
            if symbol not in ex.markets:
                logger.debug(f"Skipping {symbol}: Not in markets")
                continue
                
            market = ex.markets[symbol]
            
            if not market.get('active', True):
                logger.debug(f"Skipping {symbol}: Market not active")
                continue
                
            base = market.get('base', '').upper()
            quote = market.get('quote', '').upper()
            
            # Skip if missing required market data
            if not base or not quote:
                logger.debug(f"Skipping {symbol}: Missing base/quote info")
                continue
            
            # Filter criteria with debug logging
            if quote not in prefer_quotes:
                logger.debug(f"Skipping {symbol}: Quote {quote} not in preferred quotes {prefer_quotes}")
                continue
                
            if base in STABLES:
                logger.debug(f"Skipping {symbol}: Base {base} is a stablecoin")
                continue
                
            if not t.get('last'):
                logger.debug(f"Skipping {symbol}: No last price")
                continue
                
            volume = t.get('quoteVolume', 0)
            if volume < min_vol_usd:
                logger.debug(f"Skipping {symbol}: Volume ${volume} < ${min_vol_usd}")
                continue
                
            if t['last'] >= max_price:
                logger.debug(f"Skipping {symbol}: Price ${t['last']} >= ${max_price}")
                continue
                
            # If we get here, symbol passes all filters
            logger.debug(f"Adding {symbol}: Price=${t['last']}, Volume=${volume}")
            picks.append((symbol, volume))
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            continue
    
    # Sort by highest volume first and take top N
    picks.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, _ in picks[:top_n]]
    
    logger.info(f"Discovered {len(result)}/{len(picks)} symbols (top {top_n} by volume): {result}")
    return result

def account_quote_balance(ex: ccxt.Exchange, quote: str = None) -> float:
    """
    Get available balance for the quote currency.
    
    Args:
        ex: Exchange instance
        quote: Quote currency (e.g., "USDT", "USD"). If None, uses default from environment.
    
    Returns:
        Available balance in quote currency, or 0.0 on error
    """
    if quote is None:
        quote = "USDT" if "binance" in ex.id.lower() else "USD"
    
    try:
        balance = ex.fetch_balance()
        free = balance.get('free', {})
        total = balance.get('total', {})
        
        # Try both uppercase and lowercase versions of the symbol
        for q in [quote, quote.upper(), quote.lower()]:
            if q in free and free[q] > 0:
                return float(free[q])
            if q in total and total[q] > 0:
                return float(total[q])
        
        return 0.0
        
    except Exception as e:
        print(f"Error fetching {quote} balance: {e}")
        return 0.0

def sized_amount(ex: ccxt.Exchange, symbol: str, notional: float, price: float) -> float:
    """
    Calculate the correct order size respecting exchange limits and precision.
    
    Args:
        ex: Exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        notional: Desired notional value in quote currency
        price: Current price of the asset
        
    Returns:
        Amount in base currency, or 0 if below minimums
    """
    try:
        # Get market info with detailed logging
        market = ex.markets.get(symbol)
        if not market:
            logger.error(f"[sized_amount] {symbol} - Market not found in exchange markets")
            return 0.0
            
        min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        
        logger.info(f"[sized_amount] {symbol} - Calculating order size")
        logger.info(f"[sized_amount] {symbol} - Notional: {notional} {symbol.split('/')[1]}, Price: {price}")
        logger.info(f"[sized_amount] {symbol} - Min cost: {min_cost}, Min amount: {min_amount}")
        
        # Calculate raw amount based on notional value
        if price <= 0:
            logger.error(f"[sized_amount] {symbol} - Invalid price: {price}")
            return 0.0
            
        amount = notional / price
        logger.info(f"[sized_amount] {symbol} - Raw amount before checks: {amount}")
        
        # Check if amount is below minimum
        if amount < min_amount:
            logger.warning(f"[sized_amount] {symbol} - Amount {amount} is below minimum {min_amount}")
            return 0.0
            
        # Check if notional value is below minimum cost
        notional_value = amount * price
        if notional_value < min_cost:
            logger.warning(f"[sized_amount] {symbol} - Notional value {notional_value} is below minimum {min_cost}")
            return 0.0
            
        # Apply precision
        precision = market.get('precision', {}).get('amount')
        if precision is None:
            logger.error(f"[sized_amount] {symbol} - Could not determine precision for amount")
            return 0.0
            
        amount = float(ex.amount_to_precision(symbol, amount))
        logger.info(f"[sized_amount] {symbol} - Final amount after precision: {amount}")
        
        if amount <= 0:
            logger.warning(f"[sized_amount] {symbol} - Final amount is zero or negative: {amount}")
            
        return amount if amount > 0 else 0.0
        
    except Exception as e:
        logger.error(f"[sized_amount] Error calculating size for {symbol}: {str(e)}", exc_info=True)
        return 0.0

def breakout_signal(ohlcv: List[List[float]], lookback: int = 14, buffer: float = 0.0005) -> Tuple[str, float]:
    """
    Generate buy/sell signals based on price breakout with volume confirmation.
    
    Args:
        ohlcv: List of [timestamp, open, high, low, close, volume] lists
        lookback: Number of candles to look back for breakout detection (default: 14)
        buffer: Price buffer to reduce false breakouts (default: 0.0005 = 0.05%)
        
    Returns:
        Tuple of (signal, price) where signal is 'buy', 'sell', or 'hold'
    """
    if len(ohlcv) < lookback + 1:
        logger.debug(f"Not enough data for breakout signal. Need {lookback + 1} candles, got {len(ohlcv)}")
        return "hold", 0.0
    
    try:
        # Extract price and volume data
        highs = [c[2] for c in ohlcv[-(lookback+1):]]
        lows = [c[3] for c in ohlcv[-(lookback+1):]]
        closes = [c[4] for c in ohlcv[-(lookback+1):]]
        volumes = [c[5] for c in ohlcv[-(lookback+1):]]
        
        last_close = closes[-1]
        prev_close = closes[-2]
        
        # Calculate average volume for volume confirmation (using a shorter lookback)
        volume_lookback = min(10, len(volumes)-1)  # Use shorter lookback for volume
        avg_volume = sum(volumes[-(volume_lookback+1):-1]) / volume_lookback if volume_lookback > 0 else volumes[-1]
        last_volume = volumes[-1]
        
        # Calculate recent price range for volatility adjustment
        recent_high = max(highs[-10:])  # Last 10 candles high
        recent_low = min(lows[-10:])    # Last 10 candles low
        price_range = recent_high - recent_low
        
        # Dynamic buffer based on recent volatility
        if price_range > 0:
            dynamic_buffer = min(0.002, max(0.0005, price_range / recent_high * 0.5))  # 0.05% to 0.2% buffer
            buffer = min(buffer, dynamic_buffer)
        
        # Debug logging
        logger.debug(f"Breakout check - Close: {last_close}, Highs: {highs[:-1]}, Lows: {lows[:-1]}")
        logger.debug(f"Volume - Last: {last_volume}, Avg: {avg_volume:.2f}, Buffer: {buffer*100:.4f}%")
        
        # Check for breakout with buffer and volume confirmation
        resistance = max(highs[:-1]) * (1 + buffer)
        support = min(lows[:-1]) * (1 - buffer)
        
        # Buy signal: Price breaks above resistance with volume confirmation
        if last_close > resistance:
            volume_ok = last_volume > avg_volume * 0.7  # Reduced volume threshold to 70%
            if volume_ok:
                # Check if candle closed in the upper half of its range
                candle_range = ohlcv[-1][2] - ohlcv[-1][3]  # high - low
                if candle_range > 0:
                    close_ratio = (ohlcv[-1][4] - ohlcv[-1][3]) / candle_range  # (close - low) / range
                    if close_ratio > 0.5:  # Closed in upper half of candle
                        logger.info(f"BUY signal: {last_close} > {resistance:.8f} (resistance) with volume {last_volume:.2f} > {avg_volume*0.7:.2f}")
                        return "buy", last_close
        
        # Sell signal: Price breaks below support with volume confirmation
        elif last_close < support:
            volume_ok = last_volume > avg_volume * 0.7  # Reduced volume threshold to 70%
            if volume_ok:
                # Check if candle closed in the lower half of its range
                candle_range = ohlcv[-1][2] - ohlcv[-1][3]  # high - low
                if candle_range > 0:
                    close_ratio = (ohlcv[-1][2] - ohlcv[-1][4]) / candle_range  # (high - close) / range
                    if close_ratio > 0.5:  # Closed in lower half of candle
                        logger.info(f"SELL signal: {last_close} < {support:.8f} (support) with volume {last_volume:.2f} > {avg_volume*0.7:.2f}")
                        return "sell", last_close
        
        # Check for strong momentum if no breakout
        price_change = (last_close - prev_close) / prev_close
        if abs(price_change) > buffer * 2:  # Strong move but not a breakout
            if price_change > 0 and last_volume > avg_volume * 1.5:
                logger.info(f"BUY signal: Strong momentum {price_change*100:.2f}% with volume {last_volume:.2f} > {avg_volume*1.5:.2f}")
                return "buy", last_close
            elif price_change < 0 and last_volume > avg_volume * 1.5:
                logger.info(f"SELL signal: Strong momentum {price_change*100:.2f}% with volume {last_volume:.2f} > {avg_volume*1.5:.2f}")
                return "sell", last_close
        
        # No signal
        return "hold", 0.0
        
    except Exception as e:
        logger.error(f"Error in breakout_signal: {e}", exc_info=True)
        return "hold", 0.0

def place_market_order(ex: ccxt.Exchange, symbol: str, side: str, amount: float) -> Dict[str, Any]:
    """
    Place a market order with comprehensive validation and error handling.
    
    Args:
        ex: Exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: 'buy' or 'sell'
        amount: Amount in base currency
        
    Returns:
        Order info dict or empty dict on failure
    """
    try:
        # Get market info for validation
        market = ex.market(symbol)
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
        
        # Ensure amount is within precision limits
        try:
            amount = float(ex.amount_to_precision(symbol, amount))
        except Exception as e:
            logger.error(f"Failed to format amount {amount} for {symbol}: {e}")
            return {}
            
        # Validate amount against minimums
        if amount <= 0:
            logger.error(f"Invalid amount {amount} for {symbol}")
            return {}
            
        if amount < min_amount:
            logger.error(f"Amount {amount} is below minimum {min_amount} for {symbol}")
            return {}
            
        # Get current price to check minimum notional
        try:
            ticker = ex.fetch_ticker(symbol)
            price = ticker['last'] if ticker and 'last' in ticker else 0
            
            if price <= 0:
                logger.error(f"Invalid price {price} for {symbol}")
                return {}
                
            notional_value = amount * price
            
            if notional_value < min_cost:
                logger.error(f"Notional value {notional_value} is below minimum {min_cost} for {symbol}")
                return {}
                
            logger.info(f"Placing {side.upper()} order: {amount} {symbol} @ ~{price} (notional: {notional_value:.8f} {symbol.split('/')[1]})")
            
            # Place the order
            order = ex.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            # Log the order
            if order and 'id' in order:
                logger.info(f"Order placed: {order['id']} - {side.upper()} {amount} {symbol}")
                log_fill(ex=ex, order=order)
                return order
                
            logger.error(f"Failed to place order: {order}")
            return {}
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error while placing order: {e}")
            return {}
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error while placing order: {e}")
            return {}
            
    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds for {side} {amount} {symbol}: {e}")
        return {}
    except ccxt.InvalidOrder as e:
        logger.error(f"Invalid order parameters for {side} {amount} {symbol}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in place_market_order: {e}", exc_info=True)
        return {}
        return {}
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error on {side} {amount} {symbol}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error placing {side} order for {amount} {symbol}: {e}", exc_info=True)
        return {}


def start_exit_monitor(
    exchange: ccxt.Exchange, 
    symbol: str, 
    side: str, 
    amount: float, 
    entry: float,
    tp_pct: float,
    sl_pct: float,
    trailing_stop: bool = True,
    trailing_deviation: float = 0.1  # 0.1% trailing stop
):
    """
    Enhanced exit monitor for scalping strategy with trailing stops and dynamic exits.
    
    Args:
        exchange: The exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: 'buy' or 'sell'
        amount: Amount in base currency
        entry: Entry price
        tp_pct: Take profit percentage (e.g., 0.0075 for 0.75%)
        sl_pct: Stop loss percentage (e.g., 0.0025 for 0.25%)
        trailing_stop: Whether to use trailing stop
        trailing_deviation: Trailing stop deviation percentage
    """
    def monitor():
        try:
            logger.info(f"Starting exit monitor for {symbol} {side.upper()} {amount} @ {entry}")
            
            # Calculate initial exit prices
            if side == 'buy':
                take_profit = entry * (1 + tp_pct)
                stop_loss = entry * (1 - sl_pct)
                highest_price = entry
            else:  # sell/short
                take_profit = entry * (1 - tp_pct)
                stop_loss = entry * (1 + sl_pct)
                lowest_price = entry
            
            logger.info(f"Initial TP: {take_profit:.8f}, SL: {stop_loss:.8f}")
            
            while True:
                try:
                    # Get current ticker
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Update trailing stop for long positions
                    if side == 'buy' and trailing_stop:
                        if current_price > highest_price:
                            highest_price = current_price
                            # Update stop loss to trail below the highest price
                            new_sl = highest_price * (1 - trailing_deviation/100)
                            stop_loss = max(stop_loss, new_sl)
                    # Update trailing stop for short positions
                    elif side == 'sell' and trailing_stop:
                        if current_price < lowest_price:
                            lowest_price = current_price
                            # Update stop loss to trail above the lowest price
                            new_sl = lowest_price * (1 + trailing_deviation/100)
                            stop_loss = min(stop_loss, new_sl)
                    
                    # Check exit conditions
                    if (side == 'buy' and (current_price >= take_profit or current_price <= stop_loss)) or \
                       (side == 'sell' and (current_price <= take_profit or current_price >= stop_loss)):
                        
                        # Determine exit side (opposite of entry)
                        exit_side = 'sell' if side == 'buy' else 'buy'
                        
                        # Place market order to exit
                        logger.info(f"Exiting {symbol} {side.upper()} at {current_price} "
                                  f"(TP: {take_profit:.8f}, SL: {stop_loss:.8f})")
                        
                        # Use IOC (Immediate or Cancel) for better execution
                        order = exchange.create_order(
                            symbol=symbol,
                            type='limit',
                            side=exit_side,
                            amount=amount,
                            price=current_price,
                            params={'timeInForce': 'IOC'}
                        )
                        
                        if order and order.get('status') == 'closed':
                            log_fill(exchange, order)
                            logger.info(f"Successfully exited {symbol} at {order['price']}")
                        else:
                            # Fallback to market order if limit fails
                            logger.warning("Limit exit failed, trying market order")
                            order = place_market_order(exchange, symbol, exit_side, amount)
                            if order:
                                log_fill(exchange, order)
                        
                        break
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in exit monitor for {symbol}: {e}")
                    time.sleep(5)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Fatal error in exit monitor for {symbol}: {e}", exc_info=True)
    
    # Start the monitor in a daemon thread
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread


def log_fill(exchange: ccxt.Exchange, order: Dict) -> None:
    """
    Log a filled order to the PnL log file.
    
    Args:
        exchange: The exchange instance
        order: The filled order details from CCXT
    """
    try:
        pnl_file = os.getenv("PNL_LOG_FILE", "trading_pnl.csv")
        file_exists = os.path.isfile(pnl_file)
        
        with open(pnl_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'timestamp', 'exchange', 'symbol', 'side', 'price',
                    'amount', 'cost', 'fee', 'fee_currency', 'status'
                ])
            
            # Calculate fee info
            fee = order.get('fee', {})
            fee_cost = fee.get('cost', 0)
            fee_currency = fee.get('currency', '')
            
            writer.writerow([
                order.get('timestamp', int(time.time() * 1000)),
                exchange.id,
                order.get('symbol', ''),
                order.get('side', '').lower(),
                order.get('price', 0),
                order.get('filled', 0),
                order.get('cost', 0),
                fee_cost,
                fee_currency,
                order.get('status', '')
            ])
            
        logger.info(f"Logged {order['side']} order for {order['symbol']} to {pnl_file}")
    except Exception as e:
        logger.error(f"Error logging fill: {e}", exc_info=True)
    except ccxt.InvalidOrder as e:
        print(f"Invalid order parameters for {side} {amount} {symbol}: {e}")
    except ccxt.ExchangeError as e:
        print(f"Exchange error on {side} {amount} {symbol}: {e}")
    except Exception as e:
        print(f"Error placing {side} order for {amount} {symbol}: {e}")
    
    return {}
