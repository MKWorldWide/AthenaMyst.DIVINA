#!/usr/bin/env python3
"""
Kraken Trading Bot with Ichimoku Cloud Strategy

This script implements a multi-timeframe Ichimoku cloud strategy for Kraken.
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
        logging.FileHandler(os.getenv('KRAKEN_LOG_FILE', 'kraken_ichimoku_trader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KrakenIchimokuTrader')

class KrakenIchimokuTrader:
    """Kraken trading bot using Ichimoku Cloud strategy."""
    
    def __init__(self):
        """Initialize the Kraken trader with Ichimoku strategy."""
        self.exchange = self._init_exchange()
        self.trading_pairs = os.getenv('KRAKEN_TRADING_PAIRS', 'SHIB/USDT,DOGE/USDT,MATIC/USDT,ADA/USDT,ALGO/USDT').split(',')
        self.timeframes = os.getenv('KRAKEN_TIMEFRAMES', '5m,15m,1h').split(',')
        self.trade_amount = float(os.getenv('KRAKEN_TRADE_AMOUNT', '50'))
        self.max_open_trades = int(os.getenv('KRAKEN_MAX_OPEN_TRADES', '5'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', '1.0')) / 100
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', '2.0')) / 100
        
        # Capital allocation settings
        self.min_hold_period = timedelta(hours=6)
        self.performance_lookback = 24 * 7  # 1 week
        self.min_profit_pct = 0.02
        self.max_portfolio_allocation = 0.3
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '5.0')) / 100
        
        # Track daily performance
        self.daily_start_balance = None
        self.daily_high_balance = None
        self.daily_low_balance = None
        self.last_balance_update = None
        
        # Track active trades and performance
        self.active_trades = {}
        self.trade_history = []
        self.starting_balance = None
        
        # Load market data for all trading pairs
        self.markets = self.exchange.load_markets()
        
        # Filter out any invalid pairs
        self.trading_pairs = [pair for pair in self.trading_pairs 
                            if pair in self.exchange.markets 
                            and self.exchange.markets[pair].get('active', False)]
        
        logger.info(f"Initialized Kraken trader with {len(self.trading_pairs)} valid trading pairs")
        
        # Set sandbox mode if in test environment
        if os.getenv('ENV') == 'test':
            self.exchange.set_sandbox_mode(True)
            
        # Track open positions
        self.open_positions = {}
        
        logger.info(f"Initialized Kraken trader with pairs: {', '.join(self.trading_pairs)}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a .env file and set up trading parameters."""
        load_dotenv(config_path)
        
        # Base configuration
        config = {
            'kraken_api_key': os.getenv('KRAKEN_API_KEY'),
            'kraken_api_secret': os.getenv('KRAKEN_API_SECRET'),
            'trading_pairs': os.getenv('TRADING_PAIRS', 'SHIB/USD,DOGE/USD,MATIC/USD,ADA/USD,ALGO/USD').split(','),
            'timeframes': os.getenv('TIMEFRAMES', '5m,15m,1h').split(','),
            'trade_amount': float(os.getenv('TRADE_AMOUNT', '25')),  # $25 per trade
            'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '10')),  # Max concurrent trades
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.02')),  # 2% stop loss
            'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.05')),  # 5% take profit
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '100')),  # $100 max per position
            'min_24h_volume': float(os.getenv('MIN_24H_VOLUME', '100000')),  # $100k minimum 24h volume
            'max_coin_price': float(os.getenv('MAX_COIN_PRICE', '5.0')),  # $5 max coin price
        }
        
        logger.info(f"Loaded configuration with {len(config['trading_pairs'])} trading pairs")
        logger.info(f"Trading parameters: ${config['trade_amount']} per trade, {config['max_open_trades']} max trades")
        
        return config
    
    def _init_exchange(self):
        """Initialize and return the Kraken exchange object with optimized settings."""
        exchange = ccxt.kraken({
            'apiKey': self.config['kraken_api_key'],
            'secret': self.config['kraken_api_secret'],
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # 60 seconds
                'defaultType': 'spot',
            },
            'timeout': 30000,  # 30 second timeout
        })
        
        # Load markets and filter for top 25 coins under $5
        self._load_and_filter_markets(exchange)
        
        return exchange
        
    def _load_and_filter_markets(self, exchange):
        """Load and filter markets to focus on top 25 coins under $5 with good liquidity."""
        logger.info("Loading markets from Kraken...")
        
        try:
            # Load all markets
            markets = exchange.load_markets()
            
            # Filter for USD pairs that are active
            usd_pairs = [
                symbol for symbol, market in markets.items() 
                if market['quote'] == 'USD' 
                and market['active'] is True
                and market['spot'] is True
            ]
            
            logger.info(f"Found {len(usd_pairs)} active USD trading pairs")
            
            # Get tickers for all USD pairs
            tickers = exchange.fetch_tickers(usd_pairs)
            
            # Filter for coins under $5 with sufficient volume
            valid_pairs = []
            for symbol, ticker in tickers.items():
                try:
                    price = ticker['last'] if ticker['last'] is not None else 0
                    volume = ticker['quoteVolume'] if ticker['quoteVolume'] is not None else 0
                    
                    if (price > 0 
                        and price <= self.config['max_coin_price'] 
                        and volume >= self.config['min_24h_volume']):
                        valid_pairs.append({
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'ticker': ticker
                        })
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
            
            # Sort by 24h volume (descending) and take top 25
            valid_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_pairs = valid_pairs[:25]
            
            # Update trading pairs in config
            self.config['trading_pairs'] = [pair['symbol'] for pair in top_pairs]
            
            logger.info(f"Selected top {len(top_pairs)} coins under ${self.config['max_coin_price']}:")
            for i, pair in enumerate(top_pairs, 1):
                logger.info(f"{i}. {pair['symbol']} - ${pair['price']:.4f} (24h vol: ${pair['volume']:,.2f})")
            
        except Exception as e:
            logger.error(f"Error loading/filtering markets: {e}")
            # Fall back to default pairs if there's an error
            self.config['trading_pairs'] = ['SHIB/USD', 'DOGE/USD', 'MATIC/USD', 'ADA/USD', 'ALGO/USD']
            logger.warning(f"Using default trading pairs: {', '.join(self.config['trading_pairs'])}")
        
        # Set sandbox mode if in test environment
        if os.getenv('ENV') == 'test':
            exchange.set_sandbox_mode(True)
            
        return exchange
    
    def get_portfolio_value(self):
        """Get the total portfolio value in USDT."""
        try:
            balances = self.exchange.fetch_balance()
            total_value = 0.0
            portfolio = {}
            
            # Get current prices for all trading pairs
            tickers = {}
            for symbol in self.trading_pairs:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    tickers[symbol] = ticker['last']
                except Exception as e:
                    logger.error(f"Error fetching ticker for {symbol}: {e}")
            
            # Calculate total value in USDT
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
    
    def analyze_coin_performance(self, symbol: str) -> Dict:
        """Analyze the performance of a specific coin."""
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
    
    def optimize_capital_allocation(self):
        """Optimize capital allocation by analyzing which coins to hold or sell."""
        try:
            # Get current portfolio value and balances
            portfolio = self.get_portfolio_value()
            if not portfolio or 'total_value' not in portfolio:
                logger.error("Failed to get portfolio value")
                return
                
            total_value = portfolio['total_value']
            
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
            logger.info(f"\n=== Kraken Capital Allocation Optimization ===")
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
    
    def place_market_order(self, symbol: str, side: str, amount: float):
        """Place a market order."""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount
            )
            logger.info(f"Placed {side} order for {amount} {symbol}: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Error placing {side} order for {amount} {symbol}: {e}")
            return None
    
    def run(self):
        """Run the trading bot."""
        logger.info("Starting Kraken Ichimoku trading bot")
        
        while True:
            try:
                # Optimize capital allocation
                self.optimize_capital_allocation()
                
                # Sleep for 1 hour before next optimization
                time.sleep(3600)
                
            except KeyboardInterrupt:
                logger.info("Stopping Kraken Ichimoku trading bot")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    """Main function to run the Kraken Ichimoku trader."""
    try:
        trader = KrakenIchimokuTrader()
        trader.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
