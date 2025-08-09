import os, time, math, threading, csv, datetime
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from dotenv import load_dotenv
import ccxt

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
    ex = ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY"),
        "secret": os.getenv("KRAKEN_API_SECRET"),
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
            "recvWindow": 60000,  # Kraken default is 60s
        },
    })
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
    if prefer_quotes is None:
        prefer_quotes = {"USDT", "USD"}
    if max_price is None:
        max_price = float(os.getenv("MAX_PRICE", "5"))
    if min_vol_usd is None:
        min_vol_usd = float(os.getenv("MIN_VOL_USD", "500000"))
    
    tickers = ex.fetch_tickers()
    picks = []
    
    for symbol, t in tickers.items():
        try:
            # Skip if market data is incomplete
            if symbol not in ex.markets or not ex.markets[symbol].get('active', True):
                continue
                
            market = ex.markets[symbol]
            base = market.get('base', '').upper()
            quote = market.get('quote', '').upper()
            
            # Filter criteria
            if (quote not in prefer_quotes or  # Only preferred quote currencies
                base in STABLES or             # No stablecoin bases
                not t.get('last') or           # Must have price
                not t.get('quoteVolume', 0) >= min_vol_usd or  # Volume threshold
                t['last'] >= max_price):       # Price threshold
                continue
                
            picks.append((symbol, t.get('quoteVolume', 0)))
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Sort by highest volume first and take top N
    picks.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in picks[:top_n]]

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
        
        print(f"\n[DEBUG] position_size({symbol}, risk_amount={risk_amount}, price={price})")
        print(f"[DEBUG] min_cost={min_cost}, min_amount={min_amount}")
        
        # Calculate raw amount based on risk
        amount = risk_amount / price
        print(f"[DEBUG] Raw amount: {amount}")
        
        # Check minimum cost
        if amount * price < min_cost:
            print(f"[DEBUG] Amount {amount * price} < min_cost {min_cost}, adjusting")
        
        # Ensure we meet minimum cost
        if price * amt < min_cost:
            amt = float(ex.amount_to_precision(symbol, (min_cost/price) * 1.02))
        
        # Final sanity check
        if price * amt < min_cost:
            return 0.0
            
        return amt
        
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
    Place a market order with error handling and logging.
    
    Args:
        ex: Exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        side: 'buy' or 'sell'
        amount: Amount in base currency
        
    Returns:
        Order info dict or empty dict on failure
    """
    try:
        # Ensure amount is within precision limits
        amount = float(ex.amount_to_precision(symbol, amount))
        if amount <= 0:
            print(f"Invalid amount {amount} for {symbol}")
            return {}
            
        order = ex.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount
        )
        
        # Log the order
        log_fill(
            ex=ex.id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=order.get('price') or order.get('average') or 0,
            order_id=order.get('id', '')
        )
        
        return order
        
    except ccxt.InsufficientFunds as e:
        print(f"Insufficient funds for {side} {amount} {symbol}: {e}")
    except ccxt.InvalidOrder as e:
        print(f"Invalid order parameters for {side} {amount} {symbol}: {e}")
    except ccxt.ExchangeError as e:
        print(f"Exchange error on {side} {amount} {symbol}: {e}")
    except Exception as e:
        print(f"Error placing {side} order for {amount} {symbol}: {e}")
    
    return {}
