import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env.optimized
load_dotenv('.env.optimized')

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv('LOG_FILE', 'optimized_bot.log'))
    ]
)
logger = logging.getLogger('OptimizedKrakenBot')

@dataclass
class TradeSignal:
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    confidence: float
    indicators: dict
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class OptimizedKrakenBot:
    def __init__(self):
        """Initialize the optimized Kraken trading bot."""
        self.exchange = self._init_exchange()
        self.config = self._load_config()
        self.open_trades = {}
        self.last_analysis = {}
        self.metrics = {
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'last_trade_time': None
        }
        
    def _init_exchange(self):
        """Initialize the Kraken exchange with proper configuration."""
        return ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': int(os.getenv('KRAKEN_RECVWINDOW', '60000')),
            },
            'timeout': int(os.getenv('REQUEST_TIMEOUT', '10000')),
        })
    
    def _load_config(self) -> dict:
        """Load configuration from environment variables."""
        return {
            'symbols': self._parse_symbols(os.getenv('KRAKEN_SYMBOLS', '')),
            'timeframe': os.getenv('TIMEFRAME', '1m'),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.03')),
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PERCENT', '0.002')),
            'take_profit_pct': float(os.getenv('TAKE_PROFIT_PERCENT', '0.008')),
            'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '15')),
            'min_volume_usd': float(os.getenv('MIN_VOL_USD', '5000000')),
            'max_price': float(os.getenv('MAX_PRICE', '5.0')),
            'enable_trailing_stop': os.getenv('ENABLE_TRAILING_STOP', 'true').lower() == 'true',
            'trailing_stop_pct': float(os.getenv('TRAILING_STOP_PCT', '0.0015')),
            'min_confidence': float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.65')),
            'position_size_multiplier': float(os.getenv('POSITION_SIZE_MULTIPLIER', '1.2')),
            'ohlcv_limit': int(os.getenv('OHLCV_LIMIT', '200')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'rate_limit_delay': float(os.getenv('RATE_LIMIT_DELAY', '0.1')),
        }
    
    def _parse_symbols(self, symbols_str: str) -> List[str]:
        """Parse symbols from environment variable."""
        if not symbols_str.strip():
            return self._discover_symbols()
        return [s.strip() for s in symbols_str.split(',') if s.strip()]
    
    def _discover_symbols(self) -> List[str]:
        """Discover tradable symbols based on volume and price filters."""
        try:
            self.exchange.load_markets()
            markets = self.exchange.markets
            
            # Filter for USD pairs with sufficient volume and price
            symbols = []
            for symbol, market in markets.items():
                if market['quote'] != 'USD' or not market['active']:
                    continue
                    
                ticker = self.exchange.fetch_ticker(symbol)
                if (ticker['quoteVolume'] >= self.config['min_volume_usd'] and 
                    ticker['last'] <= self.config['max_price']):
                    symbols.append((symbol, ticker['quoteVolume']))
            
            # Sort by volume and take top 10
            symbols.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in symbols[:10]]
            
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return ['BTC/USD', 'ETH/USD', 'SOL/USD']  # Fallback
    
    def run(self):
        """Main trading loop."""
        logger.info("Starting optimized Kraken trading bot")
        
        while True:
            try:
                self._check_daily_reset()
                
                for symbol in self.config['symbols']:
                    try:
                        # Check if we can take more trades
                        if len(self.open_trades) >= self.config['max_open_trades']:
                            logger.debug("Max open trades reached, skipping symbol check")
                            continue
                            
                        # Skip if in cooldown
                        if symbol in self.open_trades:
                            continue
                            
                        # Generate and execute signal if valid
                        signal = self.analyze_market(symbol)
                        if signal and signal.confidence >= self.config['min_confidence']:
                            self.execute_trade(signal)
                            
                        # Respect rate limits
                        time.sleep(self.config['rate_limit_delay'])
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        time.sleep(5)
                
                # Sleep between full scans
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested, closing all positions...")
                self.close_all_positions()
                break
                
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(30)
    
    def analyze_market(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze market conditions and generate trading signals."""
        try:
            # Fetch OHLCV data
            df = self._fetch_ohlcv(symbol)
            if df.empty or len(df) < 100:  # Need sufficient data
                return None
                
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Get current price and indicators
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Generate signal
            signal = self._generate_signal(symbol, df)
            if not signal:
                return None
                
            logger.info(f"Generated {signal.side.upper()} signal for {symbol} "
                       f"with {signal.confidence:.2f} confidence")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data with retries."""
        for attempt in range(self.config['max_retries']):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.config['timeframe'],
                    limit=self.config['ohlcv_limit']
                )
                
                if not ohlcv or len(ohlcv) < 2:
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    dtype='float32'
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                return df
                
            except Exception as e:
                if attempt == self.config['max_retries'] - 1:
                    logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
                    return pd.DataFrame()
                time.sleep(1)
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # VWAP
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['vwap'] = vwap
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(365)
        
        return df.dropna()
    
    def _generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal based on technical analysis."""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Basic trend detection
        price_above_vwap = current['close'] > current['vwap']
        rsi_oversold = current['rsi'] < 30
        rsi_overbought = current['rsi'] > 70
        volume_ok = current['volume_ratio'] > 1.2
        
        # Signal generation
        signal = None
        confidence = 0.0
        
        # Buy signal
        if price_above_vwap and rsi_oversold and volume_ok:
            signal = 'buy'
            confidence = 0.8
        # Sell signal (for shorting)
        elif not price_above_vwap and rsi_overbought and volume_ok:
            signal = 'sell'
            confidence = 0.8
        
        if not signal:
            return None
            
        return TradeSignal(
            symbol=symbol,
            side=signal,
            price=current['close'],
            confidence=confidence,
            indicators={
                'rsi': current['rsi'],
                'vwap': current['vwap'],
                'volume_ratio': current['volume_ratio'],
                'volatility': current['volatility']
            }
        )
    
    def execute_trade(self, signal: TradeSignal):
        """Execute a trade based on the signal."""
        try:
            # Skip if already in a trade for this symbol
            if signal.symbol in self.open_trades:
                return
                
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                return
                
            # Place order
            order = self.exchange.create_market_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=position_size
            )
            
            if order and order.get('status') == 'closed':
                # Record the trade
                self.open_trades[signal.symbol] = {
                    'side': signal.side,
                    'amount': position_size,
                    'entry_price': signal.price,
                    'entry_time': time.time(),
                    'stop_loss': self._calculate_stop_loss(signal),
                    'take_profit': self._calculate_take_profit(signal),
                    'trailing_stop': self.config['enable_trailing_stop'],
                    'highest_price': signal.price if signal.side == 'buy' else float('inf'),
                    'lowest_price': signal.price if signal.side == 'sell' else 0
                }
                
                logger.info(f"Executed {signal.side.upper()} {position_size} {signal.symbol} @ {signal.price:.8f}")
                
                # Update metrics
                self.metrics['trades_today'] += 1
                self.metrics['last_trade_time'] = time.time()
                
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            quote_currency = signal.symbol.split('/')[1]
            free_balance = balance.get(quote_currency, {}).get('free', 0)
            
            if free_balance <= 0:
                logger.warning(f"Insufficient {quote_currency} balance")
                return 0.0
                
            # Calculate position size based on risk
            risk_amount = free_balance * self.config['risk_per_trade']
            position_size = (risk_amount / (signal.price * self.config['stop_loss_pct'])) * self.config['position_size_multiplier']
            
            # Get market info for precision
            market = self.exchange.market(signal.symbol)
            if market:
                # Round to allowed precision
                precision = market['precision']['amount']
                position_size = round(position_size, precision)
                
                # Ensure minimum notional
                min_notional = float(market['limits']['cost'].get('min', 0))
                if position_size * signal.price < min_notional:
                    position_size = (min_notional * 1.1) / signal.price
                    position_size = round(position_size, precision)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_stop_loss(self, signal: TradeSignal) -> float:
        """Calculate stop loss price."""
        if signal.side == 'buy':
            return signal.price * (1 - self.config['stop_loss_pct'])
        else:  # sell/short
            return signal.price * (1 + self.config['stop_loss_pct'])
    
    def _calculate_take_profit(self, signal: TradeSignal) -> float:
        """Calculate take profit price."""
        if signal.side == 'buy':
            return signal.price * (1 + self.config['take_profit_pct'])
        else:  # sell/short
            return signal.price * (1 - self.config['take_profit_pct'])
    
    def monitor_positions(self):
        """Monitor open positions and manage exits."""
        for symbol, position in list(self.open_trades.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Update highest/lowest price for trailing stop
                if position['side'] == 'buy':
                    position['highest_price'] = max(position['highest_price'], current_price)
                else:  # sell/short
                    position['lowest_price'] = min(position['lowest_price'], current_price)
                
                # Check exit conditions
                exit_reason = None
                
                # Take profit hit
                if ((position['side'] == 'buy' and current_price >= position['take_profit']) or
                    (position['side'] == 'sell' and current_price <= position['take_profit'])):
                    exit_reason = 'take_profit'
                
                # Stop loss hit
                elif ((position['side'] == 'buy' and current_price <= position['stop_loss']) or
                      (position['side'] == 'sell' and current_price >= position['stop_loss'])):
                    exit_reason = 'stop_loss'
                
                # Trailing stop
                elif position['trailing_stop']:
                    trail_amount = position['highest_price'] * self.config['trailing_stop_pct']
                    new_stop = position['highest_price'] - trail_amount
                    
                    if position['side'] == 'buy' and new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    # Check if current price hit the trailing stop
                    if current_price <= position['stop_loss']:
                        exit_reason = 'trailing_stop'
                
                # Time-based exit (optional)
                elif time.time() - position['entry_time'] > 3600:  # 1 hour max
                    exit_reason = 'timeout'
                
                # Exit the position if needed
                if exit_reason:
                    self._exit_position(symbol, position, current_price, exit_reason)
                
            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")
    
    def _exit_position(self, symbol: str, position: dict, exit_price: float, reason: str):
        """Exit a position."""
        try:
            # Determine exit side (opposite of entry)
            exit_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            # Place exit order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=exit_side,
                amount=position['amount']
            )
            
            if order and order.get('status') == 'closed':
                # Calculate PnL
                entry_price = position['entry_price']
                if position['side'] == 'buy':
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # sell/short
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                
                # Update metrics
                self.metrics['pnl'] += pnl_pct
                if pnl_pct > 0:
                    self.metrics['wins'] += 1
                else:
                    self.metrics['losses'] += 1
                
                logger.info(
                    f"Exited {position['side'].upper()} {position['amount']} {symbol} @ {exit_price:.8f} "
                    f"(P&L: {pnl_pct:.2f}%, Reason: {reason})"
                )
                
                # Remove from open trades
                self.open_trades.pop(symbol, None)
                
        except Exception as e:
            logger.error(f"Error exiting position {symbol}: {e}")
    
    def close_all_positions(self):
        """Close all open positions."""
        for symbol in list(self.open_trades.keys()):
            try:
                position = self.open_trades[symbol]
                ticker = self.exchange.fetch_ticker(symbol)
                self._exit_position(symbol, position, ticker['last'], 'shutdown')
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
    
    def _check_daily_reset(self):
        """Reset daily metrics if it's a new day."""
        now = datetime.utcnow()
        if not hasattr(self, '_last_reset_date'):
            self._last_reset_date = now.date()
        
        if now.date() > self._last_reset_date:
            logger.info("New day, resetting daily metrics")
            self.metrics.update({
                'trades_today': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0
            })
            self._last_reset_date = now.date()
    
    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            'status': 'running',
            'open_trades': len(self.open_trades),
            'metrics': self.metrics,
            'last_analysis': self.last_analysis,
            'config': self.config
        }

if __name__ == "__main__":
    try:
        bot = OptimizedKrakenBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Shutdown complete")
