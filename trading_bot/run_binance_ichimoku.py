#!/usr/bin/env python3
"""
Binance Trading Bot with Ichimoku Cloud Strategy

This script implements a multi-timeframe Ichimoku cloud strategy for Binance.
"""
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import ccxt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from trading_bot.ichimoku_strategy import generate_trading_signals, analyze_ichimoku
from trading_bot.serafina_integration import report_trade_to_serafina

# Load environment variables
load_dotenv('.env.live')

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'binance_ichimoku_trader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BinanceIchimokuTrader')

class BinanceIchimokuTrader:
    """Binance trading bot using Ichimoku Cloud strategy."""
    
    def _create_session(self):
        """Create a requests session with forced IPv4 and retry logic."""
        import requests
        from requests.adapters import HTTPAdapter
        import socket
        import urllib3
        
        # Monkey patch getaddrinfo to force IPv4
        original_getaddrinfo = socket.getaddrinfo
        
        def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
            return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        
        # Apply the monkey patch
        socket.getaddrinfo = getaddrinfo_ipv4
        
        # Create a session with custom settings
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = urllib3.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        # Create a custom adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        # Mount the custom adapter
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Set timeouts
        session.timeout = 30
        
        return session
        
    def __init__(self):
        """Initialize the Binance trader with Ichimoku strategy."""
        self.exchange = self._init_exchange()
        self.trading_pairs = os.getenv('TRADING_PAIRS', 'SHIB/USDT,DOGE/USDT,MATIC/USDT,ADA/USDT,ALGO/USDT').split(',')
        self.timeframes = os.getenv('TIMEFRAMES', '5m,15m,1h').split(',')
        self.trade_amount = float(os.getenv('TRADE_AMOUNT', '50'))
        self.max_open_trades = int(os.getenv('MAX_OPEN_TRADES', '5'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', '1.0')) / 100
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', '2.0')) / 100
        
        # Capital allocation settings
        self.min_hold_period = timedelta(hours=6)  # Minimum time to hold a position before considering selling
        self.performance_lookback = 24 * 7  # Lookback period in hours for performance analysis
        self.min_profit_pct = 0.02  # Minimum profit percentage to consider keeping a position
        self.max_portfolio_allocation = 0.3  # Maximum allocation to a single coin (30%)
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '5.0')) / 100
        self.slippage_pct = float(os.getenv('TRADE_SLIPPAGE', '0.1')) / 100
        
        # Ichimoku parameters
        self.ichimoku_params = {
            'tenkan': int(os.getenv('ICHIMOKU_TENKAN', '9')),
            'kijun': int(os.getenv('ICHIMOKU_KIJUN', '26')),
            'senkou_b': int(os.getenv('ICHIMOKU_SENKOU_B', '52')),
            'displacement': int(os.getenv('ICHIMOKU_DISPLACEMENT', '26'))
        }
        
        # Track open positions and daily P&L
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.last_daily_reset = datetime.now().date()
        
        # Load markets
        self.exchange.load_markets()
        
        logger.info("Binance Ichimoku Trader initialized")
        logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")
        logger.info(f"Timeframes: {', '.join(self.timeframes)}")
        logger.info(f"Trade amount: ${self.trade_amount}")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize and return the Binance.US exchange instance."""
        exchange = ccxt.binanceus({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Spot trading
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # 1 minute
                'warnOnFetchOpenOrdersWithoutSymbol': False,
            },
            'urls': {
                'api': {
                    'public': 'https://api.binance.us/api/v3',
                    'private': 'https://api.binance.us/api/v3',
                }
            },
            'session': self._create_session()
        })
        
        # Test API connectivity
        try:
            exchange.fetch_balance()
            logger.info("Successfully connected to Binance API")
        except Exception as e:
            logger.error(f"Error connecting to Binance API: {e}")
            raise
            
        return exchange
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol and timeframe."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No OHLCV data returned for {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {e}")
            return None
    
    def check_daily_reset(self):
        """Reset daily P&L tracking at the start of a new day."""
        now = datetime.now().date()
        if now > self.last_daily_reset:
            logger.info("Resetting daily P&L tracking")
            self.daily_pnl = 0.0
            self.daily_start_balance = self.get_balance('USDT')
            self.last_daily_reset = now
    
    def get_balance(self, currency: str) -> float:
        """Get available balance for a currency."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(currency, {}).get('free', 0.0))
        except Exception as e:
            logger.error(f"Error fetching {currency} balance: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk parameters with proper minimum amount handling."""
        try:
            # Get account balance in USDT
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            
            if usdt_balance <= 0:
                logger.warning(f"Insufficient USDT balance: {usdt_balance}")
                return 0.0
            
            # Calculate position size based on risk per trade (1% of balance)
            risk_amount = usdt_balance * 0.01
            
            # Get market info for minimum order size and precision
            market = self.exchange.market(symbol)
            min_amount = float(market.get('limits', {}).get('amount', {}).get('min', 0))
            min_cost = float(market.get('limits', {}).get('cost', {}).get('min', 10))  # Default min cost $10
            
            # Calculate position size in quote currency (USDT)
            position_size_quote = min(risk_amount, usdt_balance * 0.05)  # Max 5% of balance per trade
            position_size = position_size_quote / price
            
            # Get precision for amount
            amount_precision = market.get('precision', {}).get('amount', 8)
            
            # Round to the correct precision
            position_size = round(position_size, amount_precision)
            
            # Ensure position size meets minimum cost requirement
            if position_size * price < min_cost:
                # If position is too small, try with minimum cost + 1% buffer
                position_size = (min_cost * 1.01) / price
                position_size = round(position_size, amount_precision)
            
            # Ensure position size meets minimum amount requirement
            if position_size < min_amount:
                position_size = min_amount
            
            # Final validation
            if position_size * price < min_cost:
                logger.warning(f"Position value ${position_size * price:.2f} for {symbol} is below minimum cost ${min_cost:.2f}")
                return 0.0
                
            if position_size < min_amount:
                logger.warning(f"Position size {position_size} for {symbol} is below minimum {min_amount}")
                return 0.0
                
            # Convert to exchange precision
            position_size = float(self.exchange.amount_to_precision(symbol, position_size))
            
            logger.info(f"Calculated position size for {symbol}: {position_size} (${position_size * price:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
            return 0.0
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Place a market order on Binance."""
        try:
            # Apply slippage adjustment
            price = self.exchange.fetch_ticker(symbol)['last']
            if side == 'buy':
                price *= (1 + self.slippage_pct)
            else:  # sell
                price *= (1 - self.slippage_pct)
            
            # Calculate amount in quote currency for logging
            quote_amount = amount * price
            
            logger.info(f"Placing {side.upper()} order for {amount} {symbol} at ~${price:.8f} (${quote_amount:.2f} total)")
            
            # Place the order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'quoteOrderQty': quote_amount}  # Use quote amount for better precision
            )
            
            # Get the filled order details
            filled_order = self.exchange.fetch_order(order['id'], symbol)
            
            # Log the filled order
            filled_amount = float(filled_order.get('filled', 0))
            avg_price = float(filled_order.get('average', price))
            cost = filled_amount * avg_price
            
            logger.info(f"Filled {side.upper()} order: {filled_amount} {symbol} at {avg_price} (${cost:.2f} total)")
            
            # Report to Serafina
            report_trade_to_serafina(
                symbol=symbol,
                side=side,
                amount=filled_amount,
                price=avg_price,
                order_id=filled_order['id'],
                exchange='binance',
                notes=f"Ichimoku Cloud Strategy - {side.upper()}"
            )
            
            return filled_order
            
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {e}")
            return None
    
    def manage_open_positions(self):
        """Check open positions and manage stop-loss/take-profit."""
        try:
            open_orders = self.exchange.fetch_open_orders()
            open_symbols = set(order['symbol'] for order in open_orders)
            
            for symbol, position in list(self.open_positions.items()):
                if symbol in open_symbols:
                    continue  # Already have an open order for this symbol
                
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = position['entry_price']
                
                # Check take profit
                if current_price >= position['take_profit']:
                    logger.info(f"Take profit hit for {symbol} at {current_price}")
                    self.close_position(symbol, 'take_profit')
                
                # Check stop loss
                elif current_price <= position['stop_loss']:
                    logger.info(f"Stop loss hit for {symbol} at {current_price}")
                    self.close_position(symbol, 'stop_loss')
                
                # Check if we should trail stop
                elif current_price > entry_price * 1.01:  # Trail after 1% profit
                    new_stop = entry_price * 1.005  # Move stop to 0.5% above entry
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        logger.info(f"Trailing stop updated for {symbol} to {new_stop}")
        
        except Exception as e:
            logger.error(f"Error managing open positions: {e}")
    
    def close_position(self, symbol: str, reason: str = 'manual'):
        """Close an open position."""
        if symbol not in self.open_positions:
            logger.warning(f"No open position found for {symbol}")
            return False
        
        position = self.open_positions[symbol]
        
        try:
            # Get current price for P&L calculation
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate P&L
            if position['side'] == 'buy':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            else:  # short position (not currently supported)
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            pnl_amount = position['amount'] * (current_price - position['entry_price'])
            
            # Place sell order
            order = self.place_market_order(symbol, 'sell', position['amount'])
            
            if order:
                # Update P&L tracking
                self.daily_pnl += pnl_amount
                
                # Log the closed position
                logger.info(f"Closed {symbol} position: {pnl_pct:.2f}% (${pnl_amount:.2f}) - {reason}")
                
                # Remove from open positions
                del self.open_positions[symbol]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing {symbol} position: {e}")
            return False
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a symbol using multi-timeframe Ichimoku strategy."""
        try:
            # Fetch OHLCV data for all timeframes
            ohlcv_data = {}
            for tf in self.timeframes:
                df = self.fetch_ohlcv_data(symbol, tf)
                if df is not None and len(df) > 0:
                    ohlcv_data[tf] = df
            
            if not ohlcv_data:
                logger.warning(f"No OHLCV data available for {symbol}")
                return {}
            
            # Generate trading signals
            signals = generate_trading_signals(ohlcv_data, **self.ichimoku_params)
            
            # Add additional analysis
            latest_tf = list(ohlcv_data.keys())[0]  # Use the first timeframe for price data
            latest_candle = ohlcv_data[latest_tf].iloc[-1]
            
            analysis = {
                'symbol': symbol,
                'price': latest_candle['close'],
                'signals': signals,
                'timestamp': datetime.now().isoformat(),
                'timeframes': list(ohlcv_data.keys())
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}
    
    def analyze_coin_performance(self, symbol: str) -> Dict:
        """Analyze the performance of a specific coin.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Get recent candles for performance analysis
            candles = self.exchange.fetch_ohlcv(
                symbol,
                '1h',
                since=int((datetime.utcnow() - timedelta(hours=self.performance_lookback)).timestamp() * 1000)
            )
            
            if not candles or len(candles) < 2:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate performance metrics
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
            volatility = df['returns'].std() * np.sqrt(24)  # Annualized
            sharpe_ratio = (df['returns'].mean() / (df['returns'].std() + 1e-9)) * np.sqrt(24)
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'last_updated': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance for {symbol}: {e}")
            return {}
    
    def get_portfolio_value(self):
        """Get the total portfolio value in USDT."""
        try:
            balances = self.exchange.fetch_balance()
            total_value = 0.0
            
            # Get current prices for all trading pairs
            tickers = {}
            for symbol in self.trading_pairs:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    tickers[symbol] = ticker['last']
                except Exception as e:
                    logger.error(f"Error fetching ticker for {symbol}: {e}")
            
            # Calculate total value in USDT
            portfolio = {}
            for symbol, price in tickers.items():
                base_currency = symbol.split('/')[0]
                if base_currency in balances and float(balances[base_currency]['free']) > 0:
                    amount = float(balances[base_currency]['free'])
                    value = amount * price
                    portfolio[base_currency] = {
                        'amount': amount,
                        'price': price,
                        'value': value
                    }
                    total_value += value
            
            # Add USDT balance
            if 'USDT' in balances:
                usdt_balance = float(balances['USDT']['free'])
                portfolio['USDT'] = {
                    'amount': usdt_balance,
                    'price': 1.0,
                    'value': usdt_balance
                }
                total_value += usdt_balance
            
            return {
                'total_value': total_value,
                'portfolio': portfolio,
                'last_updated': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return {'total_value': 0, 'portfolio': {}, 'last_updated': datetime.utcnow()}
    
    def optimize_capital_allocation(self):
        """Optimize capital allocation by analyzing which coins to hold or sell."""
        try:
            # Get current portfolio value and balances
            portfolio = self.get_portfolio_value()
            if not portfolio or 'total_value' not in portfolio:
                logger.error("Failed to get portfolio value")
                return
                
            total_value = portfolio['total_value']
            balances = {}
            for symbol, data in portfolio['portfolio'].items():
                if symbol != 'USDT':
                    balances[symbol] = {
                        'free': data['amount'],
                        'used': 0.0,
                        'total': data['amount']
                    }
            
            # Get performance data for all trading pairs with error handling
            performance_data = []
            for symbol in self.trading_pairs:
                try:
                    perf = self.analyze_coin_performance(symbol)
                    if perf:
                        performance_data.append(perf)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            if not performance_data:
                logger.warning("No performance data available for optimization")
                return
            
            # Convert to DataFrame for analysis
            perf_df = pd.DataFrame(performance_data)
            
            # Rank coins by Sharpe ratio (higher is better)
            perf_df['sharpe_rank'] = perf_df['sharpe_ratio'].rank(ascending=False)
            
            # Rank coins by volatility (lower is better)
            perf_df['volatility_rank'] = perf_df['volatility'].rank(ascending=True)
            
            # Calculate combined score (70% Sharpe, 30% inverse volatility)
            perf_df['score'] = (0.7 * perf_df['sharpe_rank'] + 0.3 * perf_df['volatility_rank'])
            
            # Sort by score (higher is better)
            perf_df = perf_df.sort_values('score', ascending=False)
            
            # Get current positions from portfolio
            positions = {}
            for symbol in self.trading_pairs:
                base_currency = symbol.split('/')[0]
                if base_currency in portfolio['portfolio'] and portfolio['portfolio'][base_currency]['amount'] > 0:
                    positions[symbol] = {
                        'amount': portfolio['portfolio'][base_currency]['amount'],
                        'value': portfolio['portfolio'][base_currency]['value']
                    }
            
            # Use the pre-calculated total portfolio value
            
            # Determine which positions to keep and which to sell
            positions_to_sell = []
            positions_to_keep = []
            
            for symbol, position in positions.items():
                # Get position performance
                perf = perf_df[perf_df['symbol'] == symbol].iloc[0] if symbol in perf_df['symbol'].values else None
                
                # Calculate current allocation
                current_alloc = position['value'] / total_value if total_value > 0 else 0
                
                # Check if we should sell
                if (perf is None or perf['score'] < perf_df['score'].median() or
                    current_alloc > self.max_portfolio_allocation or
                    (perf is not None and perf['total_return'] < -0.05)):  # 5% loss threshold
                    positions_to_sell.append(symbol)
                else:
                    positions_to_keep.append(symbol)
            
            # Log optimization results
            logger.info(f"\n=== Capital Allocation Optimization ===")
            logger.info(f"Total portfolio value: ${total_value:.2f}")
            logger.info(f"Positions to keep: {', '.join(positions_to_keep) if positions_to_keep else 'None'}")
            logger.info(f"Positions to sell: {', '.join(positions_to_sell) if positions_to_sell else 'None'}")
            logger.info("\nPerformance Analysis:")
            logger.info(perf_df[['symbol', 'total_return', 'volatility', 'sharpe_ratio', 'score']].to_string())
            
            # Execute sell orders for positions to exit
            for symbol in positions_to_sell:
                try:
                    logger.info(f"Selling {symbol} for capital reallocation")
                    self.place_market_order(symbol, 'sell', positions[symbol]['amount'])
                except Exception as e:
                    logger.error(f"Error selling {symbol}: {e}")
            
            return {
                'positions_to_keep': positions_to_keep,
                'positions_to_sell': positions_to_sell,
                'performance_data': perf_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error optimizing capital allocation: {e}")
            return {}
    
    def execute_strategy(self):
        """Execute the Ichimoku trading strategy."""
        logger.info("Starting Ichimoku trading strategy")
        
        while True:
            try:
                # Check if we need to reset daily tracking
                self.check_daily_reset()
                
                # Check for existing positions and manage them
                self.manage_open_positions()
                
                # Skip if we've reached max open trades
                if len(self.open_positions) >= self.max_open_trades:
                    logger.info(f"Max open trades reached ({len(self.open_positions)}/{self.max_open_trades}). Waiting...")
                    time.sleep(60)  # Wait a minute before checking again
                    continue
                
                # Check daily loss limit
                if self.daily_pnl < -abs(self.daily_start_balance * self.max_daily_loss_pct):
                    logger.warning(f"Daily loss limit reached (${self.daily_pnl:.2f}). Stopping trading for today.")
                    time.sleep(3600)  # Wait an hour before checking again
                    continue
                
                # Analyze each trading pair
                for symbol in self.trading_pairs:
                    try:
                        # Skip if we already have an open position
                        if symbol in self.open_positions:
                            continue
                        
                        # Analyze the symbol
                        analysis = self.analyze_symbol(symbol)
                        if not analysis:
                            continue
                        
                        # Check for buy signal
                        if analysis['signals'].get('strong_buy', False) or analysis['signals'].get('buy', False):
                            # Calculate position size
                            amount = self.calculate_position_size(symbol, analysis['price'])
                            if amount <= 0:
                                logger.warning(f"Insufficient balance to trade {symbol}")
                                continue
                            
                            # Place buy order
                            order = self.place_market_order(symbol, 'buy', amount)
                            
                            if order:
                                # Calculate stop loss and take profit levels
                                entry_price = float(order.get('average', analysis['price']))
                                stop_loss = entry_price * (1 - self.stop_loss_pct)
                                take_profit = entry_price * (1 + self.take_profit_pct)
                                
                                # Record the position
                                self.open_positions[symbol] = {
                                    'side': 'buy',
                                    'amount': amount,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                logger.info(f"Opened position: {symbol} {amount} @ {entry_price} | SL: {stop_loss:.8f} | TP: {take_profit:.8f}")
                                
                                # Check if we've reached max open trades
                                if len(self.open_positions) >= self.max_open_trades:
                                    logger.info("Max open trades reached. Moving to next cycle.")
                                    break
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        time.sleep(5)  # Small delay to avoid rate limits
                
                # Wait before next analysis cycle
                time.sleep(30)  # 30 seconds between analysis cycles
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt. Shutting down...")
                break
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    """Main function to run the Binance Ichimoku trader."""
    try:
        # Initialize and run the trader
        trader = BinanceIchimokuTrader()
        trader.execute_strategy()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
