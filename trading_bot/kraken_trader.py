import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kraken_trader.log')
    ]
)
logger = logging.getLogger('KrakenTrader')

class KrakenTrader:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the Kraken trader with API credentials."""
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # 60 seconds
            }
        })
        
        # Trading parameters
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% risk per trade
        self.take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.0075"))  # 0.75%
        self.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.0025"))  # 0.25%
        self.max_open_trades = int(os.getenv("MAX_OPEN_TRADES", "5"))
        self.min_confidence = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.7"))
        
        # State tracking
        self.open_trades: Dict[str, dict] = {}
        self.cooldown_until: Dict[str, float] = {}
        self.last_trade_time: Dict[str, float] = {}
        
        # Load markets on init
        self.exchange.load_markets()
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data with error handling and retries."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    params={'trades': True}
                )
                
                if not ohlcv or len(ohlcv) < 2:
                    logger.warning(f"Insufficient data for {symbol}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    dtype='float32'
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                return df
                
            except ccxt.NetworkError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Network error fetching {symbol}: {e}")
                    raise
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        if df.empty:
            return df
            
        # Moving Averages
        df['sma20'] = ta.sma(df['close'], length=20)
        df['sma50'] = ta.sma(df['close'], length=50)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Volume indicators
        df['volume_sma20'] = ta.sma(df['volume'], length=20)
        df['volume_spike'] = df['volume'] > (df['volume_sma20'] * 1.5)
        
        # Bollinger Bands
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df = df.join(bollinger)
        
        return df.dropna()
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[Optional[str], float]:
        """Generate trading signal with confidence score."""
        if df.empty or len(df) < 50:  # Need at least 50 periods for reliable signals
            return None, 0.0
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend detection
        trend_up = current['sma20'] > current['sma50']
        price_above_sma = current['close'] > current['sma20']
        
        # Momentum
        rsi = current['rsi']
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # Volume confirmation
        volume_ok = current['volume_spike']
        
        # Generate signal with confidence
        signal = None
        confidence = 0.0
        
        # Buy signal: Uptrend with oversold RSI and volume confirmation
        if trend_up and price_above_sma and rsi_oversold and volume_ok:
            signal = 'buy'
            confidence = 0.8
        # Sell signal: Downtrend with overbought RSI and volume confirmation
        elif not trend_up and not price_above_sma and rsi_overbought and volume_ok:
            signal = 'sell'
            confidence = 0.8
            
        return signal, confidence
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Get account balance in quote currency (e.g., USD)
            balance = self.exchange.fetch_balance()
            quote_currency = symbol.split('/')[1]
            free_balance = balance.get(quote_currency, {}).get('free', 0)
            
            if free_balance <= 0:
                logger.warning(f"Insufficient {quote_currency} balance")
                return 0.0
                
            # Calculate position size based on risk per trade
            risk_amount = free_balance * self.risk_per_trade
            stop_loss_amount = price * self.stop_loss_pct
            
            if stop_loss_amount <= 0:
                return 0.0
                
            position_size = risk_amount / stop_loss_amount
            
            # Get market info for lot size precision
            market = self.exchange.market(symbol)
            if market:
                # Round to allowed precision
                precision = market['precision']['amount']
                position_size = round(position_size, precision)
                
                # Ensure position size meets minimum notional
                min_notional = float(market['limits']['cost'].get('min', 0))
                if position_size * price < min_notional:
                    logger.warning(f"Position size too small for {symbol}")
                    return 0.0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def execute_trade(self, symbol: str, signal: str, price: float):
        """Execute a trade with proper risk management."""
        try:
            # Check if we already have an open position
            if symbol in self.open_trades:
                logger.info(f"Already have an open position in {symbol}")
                return
                
            # Calculate position size
            amount = self.calculate_position_size(symbol, price)
            if amount <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return
                
            # Place the order
            side = 'buy' if signal == 'buy' else 'sell'
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={'timeInForce': 'IOC'}
            )
            
            if order and order.get('status') == 'closed':
                # Start exit monitoring in a separate thread
                self.start_exit_monitor(symbol, side, amount, price)
                
                # Update state
                self.open_trades[symbol] = {
                    'side': side,
                    'amount': amount,
                    'entry_price': price,
                    'timestamp': time.time()
                }
                
                logger.info(f"Executed {side.upper()} {amount} {symbol} @ {price}")
                
        except Exception as e:
            logger.error(f"Error executing {signal} order for {symbol}: {e}")
    
    def start_exit_monitor(self, symbol: str, side: str, amount: float, entry_price: float):
        """Start monitoring an open position for exit conditions."""
        # This would typically run in a separate thread
        # Implementation would monitor price and manage exits
        pass
    
    def run(self):
        """Main trading loop."""
        logger.info("Starting Kraken trading bot")
        
        while True:
            try:
                # Get symbols to trade (simplified - would come from config)
                symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD']
                
                for symbol in symbols:
                    try:
                        # Skip if in cooldown
                        if self.is_in_cooldown(symbol):
                            continue
                            
                        # Fetch and process market data
                        df = self.fetch_ohlcv_data(symbol)
                        if df.empty:
                            continue
                            
                        # Add indicators
                        df = self.add_technical_indicators(df)
                        
                        # Generate signal
                        signal, confidence = self.generate_signal(df, symbol)
                        
                        # Execute trade if signal is strong enough
                        if signal and confidence >= self.min_confidence:
                            current_price = df['close'].iloc[-1]
                            self.execute_trade(symbol, signal, current_price)
                            
                        # Be nice to the API
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        time.sleep(5)
                
                # Sleep between full symbol scans
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
                
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(30)

if __name__ == "__main__":
    # Load API keys from environment
    api_key = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in environment")
    
    trader = KrakenTrader(api_key, api_secret)
    trader.run()
