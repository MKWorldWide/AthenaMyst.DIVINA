import os
import json
import time
import ccxt
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingEngine')

class TradingEngine:
    def __init__(self, exchange_id='binance', testnet=True):
        """Initialize the trading engine with exchange connection."""
        load_dotenv()
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # for futures trading
                'adjustForTimeDifference': True,
            },
        })
        
        # Set testnet if needed
        if testnet and exchange_id == 'binance':
            self.exchange.set_sandbox_mode(True)
        
        # Trading parameters
        self.trading_pair = os.getenv('TRADING_PAIR', 'EUR/USDT')
        self.base_currency = self.trading_pair.split('/')[0]
        self.quote_currency = self.trading_pair.split('/')[1]
        self.trade_amount = float(os.getenv('TRADE_AMOUNT', 100))
        self.leverage = int(os.getenv('LEVERAGE', 10))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', 2.0)) / 100
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', 4.0)) / 100
        self.max_trades = int(os.getenv('MAX_TRADES', 3))
        
        # Initialize indicators
        self.rsi_period = 14
        self.ema_fast = 12
        self.ema_slow = 26
        self.signal = 9
        self.atr_period = 14
        
        logger.info(f"Trading Engine initialized for {self.trading_pair}")
    
    async def get_historical_data(self, timeframe='1h', limit=100):
        """Fetch historical OHLCV data."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                self.trading_pair,
                timeframe=timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        try:
            # RSI
            rsi_indicator = RSIIndicator(close=df['close'], window=self.rsi_period)
            df['rsi'] = rsi_indicator.rsi()
            
            # EMAs
            ema_fast = EMAIndicator(close=df['close'], window=self.ema_fast)
            ema_slow = EMAIndicator(close=df['close'], window=self.ema_slow)
            df['ema_fast'] = ema_fast.ema_indicator()
            df['ema_slow'] = ema_slow.ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'], window_slow=self.ema_slow, 
                       window_fast=self.ema_fast, window_sign=self.signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ATR for volatility
            atr = AverageTrueRange(high=df['high'], low=df['low'], 
                                 close=df['close'], window=self.atr_period)
            df['atr'] = atr.average_true_range()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def generate_signal(self, df):
        """Generate trading signal based on indicators."""
        try:
            # Get the latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize signal
            signal = {
                'pair': self.trading_pair,
                'action': 'HOLD',
                'price': float(latest['close']),
                'timestamp': latest.name.isoformat(),
                'indicators': {
                    'rsi': float(latest['rsi']),
                    'ema_fast': float(latest['ema_fast']),
                    'ema_slow': float(latest['ema_slow']),
                    'macd': float(latest['macd']),
                    'macd_signal': float(latest['macd_signal']),
                    'atr': float(latest['atr'])
                },
                'stop_loss': None,
                'take_profit': None
            }
            
            # Check for buy signal (EMA crossover and RSI not overbought)
            if (latest['ema_fast'] > latest['ema_slow'] and 
                prev['ema_fast'] <= prev['ema_slow'] and
                latest['rsi'] < 70):
                signal['action'] = 'BUY'
                signal['stop_loss'] = signal['price'] * (1 - self.stop_loss_pct)
                signal['take_profit'] = signal['price'] * (1 + self.take_profit_pct)
            
            # Check for sell signal (EMA crossunder and RSI not oversold)
            elif (latest['ema_fast'] < latest['ema_slow'] and 
                  prev['ema_fast'] >= prev['ema_slow'] and
                  latest['rsi'] > 30):
                signal['action'] = 'SELL'
                signal['stop_loss'] = signal['price'] * (1 + self.stop_loss_pct)
                signal['take_profit'] = signal['price'] * (1 - self.take_profit_pct)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    async def execute_trade(self, signal):
        """Execute trade based on signal."""
        if signal['action'] == 'HOLD':
            return None
            
        try:
            # Get current positions
            positions = await self.exchange.fetch_positions([self.trading_pair])
            current_position = next((p for p in positions if p['symbol'] == self.trading_pair), None)
            
            # Check if we already have an open position
            if current_position and abs(float(current_position['contracts'])) > 0:
                logger.info(f"Position already open for {self.trading_pair}")
                return None
            
            # Calculate position size
            price = signal['price']
            amount = self.trade_amount / price
            
            # Place order
            order_side = 'buy' if signal['action'] == 'BUY' else 'sell'
            order_type = 'market'  # or 'limit' with a price
            
            order = await self.exchange.create_order(
                symbol=self.trading_pair,
                type=order_type,
                side=order_side,
                amount=amount,
                params={
                    'leverage': self.leverage,
                    'stopLossPrice': signal['stop_loss'],
                    'takeProfitPrice': signal['take_profit']
                }
            )
            
            logger.info(f"Trade executed: {order_side.upper()} {amount} {self.trading_pair} at {price}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    async def run_strategy(self):
        """Run the trading strategy."""
        try:
            # Get historical data
            df = await self.get_historical_data()
            if df is None:
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                return None
            
            # Generate signal
            signal = self.generate_signal(df)
            if signal is None:
                return None
            
            # Execute trade if signal is not HOLD
            if signal['action'] != 'HOLD':
                trade_result = await self.execute_trade(signal)
                signal['trade_result'] = trade_result is not None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            return None
