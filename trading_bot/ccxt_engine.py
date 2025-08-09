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
        market = ex.markets[symbol]
        min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        
        logger.debug(f"[sized_amount] {symbol} - notional: {notional}, price: {price}")
        logger.debug(f"[sized_amount] {symbol} - min_cost: {min_cost}, min_amount: {min_amount}")
        
        # Calculate raw amount based on notional value
        if price <= 0:
            logger.error(f"Invalid price {price} for {symbol}")
            return 0.0
            
        amount = notional / price
        logger.debug(f"[sized_amount] {symbol} - raw amount: {amount}")
        
        # Check if amount is below minimum
        if amount < min_amount:
            logger.warning(f"Amount {amount} is below minimum {min_amount} for {symbol}")
            return 0.0
            
        # Check if notional value is below minimum cost
        notional_value = amount * price
        if notional_value < min_cost:
            logger.warning(f"Notional value {notional_value} is below minimum {min_cost} for {symbol}")
            return 0.0
            
        # Apply precision
        precision = market['precision']['amount']
        amount = float(ex.amount_to_precision(symbol, amount))
        logger.debug(f"[sized_amount] {symbol} - final amount after precision: {amount}")
        
        return amount if amount > 0 else 0.0
        
    except Exception as e:
        print(f"Error in sized_amount for {symbol}: {e}")
        return 0.0

def breakout_signal(ohlcv: List[List[float]], lookback: int = 20, buffer: float = 0.0005) -> Tuple[str, float]:
    """
    Generate buy/sell signals based on price breakout with buffer to reduce noise.
    
    Args:
        ohlcv: List of [timestamp, open, high, low, close, volume] lists
        lookback: Number of candles to look back for breakout detection (default: 20)
        buffer: Price buffer to reduce false breakouts (default: 0.0005 = 0.05%)
        
    Returns:
        Tuple of (signal, price) where signal is 'buy', 'sell', or 'hold'
    """
    if len(ohlcv) < lookback + 1:
        return "hold", 0.0
        
    highs = [c[2] for c in ohlcv[-(lookback+1):]]  # High prices
    lows = [c[3] for c in ohlcv[-(lookback+1):]]    # Low prices
    last = ohlcv[-1][4]                             # Last close price
    
    # Check for breakout with buffer
    if last > max(highs[:-1]) * (1 + buffer):
        return "buy", last
    elif last < min(lows[:-1]) * (1 - buffer):
        return "sell", last
    
    return "hold", last

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
