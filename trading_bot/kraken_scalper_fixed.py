#!/usr/bin/env python3
"""
Optimized Kraken Scalping Bot with improved error handling and connection management.
"""
import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings
from dataclasses import dataclass, asdict
import json
from dotenv import load_dotenv
import os
from discord_webhook import DiscordWebhook

# Load environment variables
load_dotenv()

# Initialize Discord webhook if enabled
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
discord_webhook = DiscordWebhook(DISCORD_WEBHOOK_URL) if DISCORD_WEBHOOK_URL else None

# Configure logging with timezone awareness
import logging.handlers

def configure_logging():
    """Configure logging with timezone awareness and proper formatting."""
    class Formatter(logging.Formatter):
        """Custom formatter that converts timestamps to local time."""
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up file handler with rotation (10MB per file, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/kraken_scalper.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter and add to handlers
    formatter = Formatter(
        '%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger('ccxt').setLevel('WARNING')  # Reduce CCXT logging
    logging.getLogger('urllib3').setLevel('WARNING')  # Reduce urllib3 logging

# Initialize logging
configure_logging()
logger = logging.getLogger('KrakenScalper')

@dataclass
class TradeSignal:
    """Data class to hold trade signal information."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    confidence: float
    indicators: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class KrakenScalper:
    def __init__(self):
        """Initialize the Kraken scalping bot."""
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
        self._last_reset_date = datetime.now().date()
    
    def _init_exchange(self):
        """Initialize and return the Kraken exchange object."""
        try:
            exchange = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_API_SECRET'),
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                    'trading_agreement': 'agree',
                },
                'timeout': 30000,
            })
            
            # Test the connection
            exchange.fetch_balance()
            logger.info("Successfully connected to Kraken")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize Kraken exchange: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and return configuration parameters."""
        return {
            'symbols': self._parse_symbols(os.getenv('KRAKEN_SYMBOLS', 'BTC/USD,ETH/USD,SOL/USD')),
            'timeframe': os.getenv('TIMEFRAME', '1m'),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.0525')),  # 5.25%
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PERCENT', '0.0025')),  # 0.25%
            'take_profit_pct': float(os.getenv('TAKE_PROFIT_PERCENT', '0.0075')),  # 0.75%
            'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '10')),
            'min_volume_usd': float(os.getenv('MIN_VOL_USD', '2000000')),
            'max_price': float(os.getenv('MAX_PRICE', '5.0')),
            'enable_trailing_stop': os.getenv('ENABLE_TRAILING_STOP', 'true').lower() == 'true',
            'trailing_stop_pct': float(os.getenv('TRAILING_STOP_PCT', '0.0015')),  # 0.15%
            'min_confidence': float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.65')),
            'position_size_multiplier': float(os.getenv('POSITION_SIZE_MULTIPLIER', '1.2')),
            'ohlcv_limit': int(os.getenv('OHLCV_LIMIT', '100')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'rate_limit_delay': float(os.getenv('RATE_LIMIT_DELAY', '0.1')),
        }
    
    def _parse_symbols(self, symbols_str: str) -> List[str]:
        """Parse and validate trading symbols."""
        symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
        return [s if '/' in s else f"{s}/USD" for s in symbols]
    
    def run(self):
        """Main trading loop with optimized scanning and confidence-based execution."""
        logger.info("🚀 Starting Kraken Scalping Bot with 72%+ Confidence Strategy")
        
        while True:
            try:
                self._check_daily_reset()
                scan_start_time = time.time()
                
                # Get all available symbols
                symbols = self.config.get('symbols', [])
                if not symbols:
                    logger.warning("⚠️ No trading symbols configured. Using default list...")
                    symbols = [
                        'BTC/USD', 'ETH/USD', 'XRP/USD', 'DOGE/USD', 'ADA/USD',
                        'SOL/USD', 'MATIC/USD', 'DOT/USD', 'LTC/USD', 'LINK/USD',
                        'UNI/USD', 'AAVE/USD', 'ATOM/USD', 'XLM/USD', 'ALGO/USD',
                        'FIL/USD', 'VET/USD', 'THETA/USD', 'EOS/USD', 'XMR/USD',
                        'XTZ/USD', 'CAKE/USD', 'AVAX/USD', 'FTM/USD', 'SUSHI/USD'
                    ]
                
                # Log current status
                logger.info(f"🔍 Scanning {len(symbols)} symbols | Open Trades: {len(self.open_trades)}/{self.config['max_open_trades']}")
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        # Check if we can open new trades
                        if len(self.open_trades) >= self.config['max_open_trades']:
                            logger.info(f"✅ Max open trades ({self.config['max_open_trades']}) reached. Monitoring existing positions...")
                            break
                            
                        # Analyze market and generate signal with 72%+ confidence
                        signal = self.analyze_market(symbol)
                        
                        # Execute trade if signal is strong enough
                        if signal and signal.confidence >= 0.72:  # Hardcoded 72% minimum
                            logger.info(f"🎯 High-Confidence Signal ({signal.confidence*100:.1f}%): {signal.side.upper()} {signal.symbol} @ {signal.price:.8f}")
                            self.execute_trade(signal)
                        
                        # Monitor open positions (with rate limiting)
                        if symbol in self.open_trades:
                            self._monitor_position(symbol)
                            
                        # Small delay between symbol processing to avoid rate limits
                        time.sleep(0.5)
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {str(e)[:200]}")
                        time.sleep(5)
                
                # Calculate dynamic sleep time to maintain consistent scan interval
                scan_duration = time.time() - scan_start_time
                sleep_time = max(1, self.config.get('scan_interval_seconds', 10) - scan_duration)
                logger.debug(f"⏱️  Scan completed in {scan_duration:.2f}s. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"🔥 Critical error in main loop: {str(e)[:200]}")
                logger.debug("Stack trace:", exc_info=True)
                time.sleep(30)  # Longer delay on critical error
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
            if df.empty or len(df) < 50:  # Need sufficient data
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
                
            logger.info(
                f"Generated {signal.side.upper()} signal for {symbol} "
                f"@ {signal.price:.8f} (Confidence: {signal.confidence:.2f})"
            )
            
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
        
        # Calculate indicators - suppress timezone warnings for VWAP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
        """Generate trading signal based on technical analysis with enhanced logging."""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize confidence score (0-1 scale) and components
        confidence = 0.0
        signal = None
        components = {}
        
        # 1. Price relative to VWAP (25% weight)
        vwap_distance_pct = ((current['close'] - current['vwap']) / current['vwap']) * 100
        vwap_score = min(1.0, max(0, abs(vwap_distance_pct) / 0.8))  # Full points at 0.8% distance
        
        if current['close'] > current['vwap']:
            confidence += 0.25 * vwap_score
            components['vwap'] = f'above by {vwap_distance_pct:.2f}% (score: {vwap_score*100:.0f}%)'
        else:
            confidence += 0.0  # Below VWAP reduces confidence for buys
            components['vwap'] = f'below by {abs(vwap_distance_pct):.2f}%'
        
        # 2. RSI Score (30% weight) - Adjusted for more signals
        if current['rsi'] < 40:  # More lenient oversold
            rsi_score = 1.0 - (current['rsi'] / 40.0)
            confidence += 0.3 * rsi_score
            components['rsi'] = f'oversold {current["rsi"]:.1f} (score: {rsi_score*100:.0f}%)'
        elif current['rsi'] > 60:  # More lenient overbought
            rsi_score = (current['rsi'] - 60.0) / 40.0
            confidence += 0.3 * rsi_score
            components['rsi'] = f'overbought {current["rsi"]:.1f} (score: {rsi_score*100:.0f}%)'
        else:
            components['rsi'] = f'neutral {current["rsi"]:.1f}'
        
        # 3. Volume Confirmation (20% weight) - More lenient
        volume_score = min(1.0, (current['volume_ratio'] - 0.7) / 0.6)  # 0.7-1.3x avg volume
        confidence += 0.2 * volume_score
        components['volume'] = f'{current["volume_ratio"]:.2f}x avg (score: {volume_score*100:.0f}%)'
        
        # 4. Recent Price Momentum (25% weight) - More sensitive
        lookback = 2  # Shorter lookback
        if len(df) > lookback:
            returns = (df['close'].iloc[-1] / df['close'].iloc[-lookback-1]) - 1
            momentum_score = min(1.0, abs(returns) / 0.008)  # 0.8% move = full points
            confidence += 0.25 * momentum_score
            components['momentum'] = f'{returns*100:.2f}% (score: {momentum_score*100:.0f}%)'
        
        # Determine signal direction with adjusted thresholds
        if confidence >= 0.65:  # Lowered from 72%
            if current['rsi'] < 45 and current['close'] > current['vwap'] and current['volume_ratio'] > 0.8:
                signal = 'buy'
            elif current['rsi'] > 55 and current['close'] < current['vwap'] and current['volume_ratio'] > 0.8:
                signal = 'sell'
        
        # Enhanced logging
        if signal:
            logger.info(f"🔍 {symbol} {signal.upper()} signal | Confidence: {confidence*100:.1f}% | "
                      f"Price: {current['close']:.6f} | {', '.join(f'{k}: {v}' for k, v in components.items())}")
        elif confidence > 0.5:  # Log near-misses for debugging
            logger.debug(f"❌ {symbol} No signal | Confidence: {confidence*100:.1f}% | "
                       f"Price: {current['close']:.6f} | {', '.join(f'{k}: {v}' for k, v in components.items())}")
        
        if not signal:
            return None
            
        # Add some randomness to confidence to avoid clustering at exact threshold
        confidence = min(0.98, confidence + (np.random.random() * 0.04))  # Add 0-4% noise
        
        return TradeSignal(
            symbol=symbol,
            side=signal,
            price=current['close'],
            confidence=confidence,
            indicators={
                'rsi': current['rsi'],
                'vwap': current['vwap'],
                'volume_ratio': current['volume_ratio'],
                'volatility': current['volatility'],
                'confidence_components': {
                    'vwap_score': vwap_distance_pct,
                    'rsi_score': current['rsi'],
                    'volume_score': volume_score,
                    'momentum_score': returns * 100 if 'returns' in locals() else 0
                }
            }
        )
    
    def _send_trade_notification(self, symbol: str, side: str, entry_price: float, 
                              stop_loss: float, take_profit: float, amount: float):
        """Send trade notification to Discord."""
        if not discord_webhook:
            return
            
        try:
            # Format the message for Discord
            direction_emoji = "🟢" if side.lower() == 'buy' else "🔴"
            symbol_display = symbol.replace('/', '')
            
            # Create a clean, readable message
            message = (
                f"## {direction_emoji} {side.upper()} {symbol_display} {direction_emoji}\n"
                f"**Entry Price**: ${entry_price:.8f}\n"
                f"**Amount**: {amount:.8f} {symbol.split('/')[0]}\n"
                f"**Stop Loss**: ${stop_loss:.8f}\n"
                f"**Take Profit**: ${take_profit:.8f}\n"
                f"**Risk/Reward**: 1:{(take_profit/entry_price - 1)/(1 - stop_loss/entry_price):.1f}"
            )
            
            # Send the notification
            discord_webhook.send_status_update(
                message=message,
                color=0x00ff00 if side.lower() == 'buy' else 0xff0000
            )
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
    
    def execute_trade(self, signal: TradeSignal):
        """Execute a trade based on the signal."""
        try:
            # Skip if already in a trade for this symbol
            if signal.symbol in self.open_trades:
                return
                
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                logger.warning(f"Insufficient balance or invalid position size for {signal.symbol}")
                return
            
            # Calculate stop loss and take profit
            stop_loss = self._calculate_stop_loss(signal)
            take_profit = self._calculate_take_profit(signal)
                
            # Place order
            order = self.exchange.create_market_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=position_size
            )
            
            if order and order.get('status') == 'closed':
                # Record the trade
                trade_info = {
                    'side': signal.side,
                    'amount': position_size,
                    'entry_price': signal.price,
                    'entry_time': time.time(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': self.config['enable_trailing_stop'],
                    'highest_price': signal.price if signal.side == 'buy' else float('inf'),
                    'lowest_price': signal.price if signal.side == 'sell' else 0
                }
                self.open_trades[signal.symbol] = trade_info
                
                # Log the trade
                logger.info(
                    f"Executed {signal.side.upper()} {position_size:.8f} {signal.symbol} "
                    f"@ {signal.price:.8f}"
                )
                
                # Send Discord notification
                self._send_trade_notification(
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    amount=position_size
                )
                
                # Update metrics
                self.metrics['trades_today'] += 1
                self.metrics['last_trade_time'] = time.time()
                
        except Exception as e:
            error_msg = f"Error executing trade for {signal.symbol}: {str(e)}"
            logger.error(error_msg)
            if discord_webhook:
                discord_webhook.send_error(error_msg)
    
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
            position_size = (risk_amount / (signal.price * self.config['stop_loss_pct']))
            position_size *= self.config['position_size_multiplier']
            
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
    
    def _monitor_position(self, symbol: str):
        """Monitor and manage an open position."""
        try:
            position = self.open_trades[symbol]
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
                if position['side'] == 'buy':
                    trail_amount = position['highest_price'] * self.config['trailing_stop_pct']
                    new_stop = position['highest_price'] - trail_amount
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                    
                    # Check if current price hit the trailing stop
                    if current_price <= position['stop_loss']:
                        exit_reason = 'trailing_stop'
                
                else:  # sell/short
                    trail_amount = position['lowest_price'] * self.config['trailing_stop_pct']
                    new_stop = position['lowest_price'] + trail_amount
                    if new_stop < position['stop_loss'] or position['stop_loss'] == 0:
                        position['stop_loss'] = new_stop
                    
                    # Check if current price hit the trailing stop
                    if current_price >= position['stop_loss']:
                        exit_reason = 'trailing_stop'
            
            # Time-based exit (optional)
            elif time.time() - position['entry_time'] > 3600:  # 1 hour max
                exit_reason = 'timeout'
            
            # Exit the position if needed
            if exit_reason:
                self._exit_position(symbol, position, current_price, exit_reason)
            
        except Exception as e:
            logger.error(f"Error monitoring position {symbol}: {e}")
    
    def _send_exit_notification(self, symbol: str, position: Dict, exit_price: float, 
                              reason: str, pnl: float, pnl_percent: float):
        """Send trade exit notification to Discord."""
        if not discord_webhook:
            return
            
        try:
            # Format the message for Discord
            direction_emoji = "🟢" if pnl >= 0 else "🔴"
            symbol_display = symbol.replace('/', '')
            
            # Format reason for display
            reason_display = {
                'take_profit': 'Take Profit',
                'stop_loss': 'Stop Loss',
                'trailing_stop': 'Trailing Stop',
                'timeout': 'Time Limit',
                'manual': 'Manual Close'
            }.get(reason, reason.title())
            
            # Create a clean, readable message
            message = (
                f"## {direction_emoji} CLOSED {symbol_display} {direction_emoji}\n"
                f"**Side**: {position['side'].upper()}\n"
                f"**Entry Price**: ${position['entry_price']:.8f}\n"
                f"**Exit Price**: ${exit_price:.8f}\n"
                f"**Amount**: {position['amount']:.8f} {symbol.split('/')[0]}\n"
                f"**P&L**: ${abs(pnl):.8f} ({abs(pnl_percent):.2f}%)\n"
                f"**Reason**: {reason_display}"
            )
            
            # Send the notification
            discord_webhook.send_trade_closed(
                pair=symbol,
                pnl=pnl,
                pnl_percent=pnl_percent,
                reason=reason_display
            )
            
        except Exception as e:
            logger.error(f"Failed to send Discord exit notification: {e}")
    
    def _exit_position(self, symbol: str, position: Dict, exit_price: float, reason: str):
        """Exit a position."""
        try:
            # Place the exit order (opposite of entry)
            exit_side = 'sell' if position['side'] == 'buy' else 'buy'
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=exit_side,
                amount=position['amount']
            )
            
            if order and order.get('status') == 'closed':
                # Calculate P&L
                entry_price = position['entry_price']
                if position['side'] == 'buy':
                    pnl = (exit_price - entry_price) * position['amount']
                else:  # sell/short
                    pnl = (entry_price - exit_price) * position['amount']
                
                pnl_percent = (pnl / (entry_price * position['amount'])) * 100
                
                # Update metrics
                self.metrics['pnl'] += pnl
                if pnl >= 0:
                    self.metrics['wins'] += 1
                else:
                    self.metrics['losses'] += 1
                
                # Log the exit
                logger.info(
                    f"Closed {position['side'].upper()} {position['amount']:.8f} {symbol} @ {exit_price:.8f} "
                    f"(P&L: {pnl:.8f} {symbol.split('/')[1]}, {pnl_percent:.2f}%) - {reason}"
                )
                
                # Send Discord notification
                self._send_exit_notification(
                    symbol=symbol,
                    position=position,
                    exit_price=exit_price,
                    reason=reason,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                )
                
        except Exception as e:
            error_msg = f"Error exiting position {symbol}: {str(e)}"
            logger.error(error_msg)
            if discord_webhook:
                discord_webhook.send_error(error_msg)
        finally:
            # Ensure the position is removed from open_trades even if there's an error
            self.open_trades.pop(symbol, None)
    
    def close_all_positions(self):
        """Close all open positions."""
        logger.info("Closing all open positions...")
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
        now = datetime.now(timezone.utc).date()
        if now > self._last_reset_date:
            logger.info("New day, resetting daily metrics")
            self.metrics.update({
                'trades_today': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0
            })
            self._last_reset_date = now

def main():
    """Main entry point for the Kraken Scalping Bot."""
    try:
        # Initialize and run the bot
        bot = KrakenScalper()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
