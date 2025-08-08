import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from divina_scalper import DivinaScalper

class EnhancedDivinaScalper(DivinaScalper):
    """Enhanced version of DivinaScalper with advanced technical analysis."""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Enhanced Scalping Strategy"
        self.setup_indicators()
    
    def setup_indicators(self):
        """Initialize indicator parameters."""
        # Trend indicators
        self.ema_fast_period = 9
        self.ema_medium_period = 21
        self.ema_slow_period = 50
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Momentum indicators
        self.rsi_period = 14
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_smooth = 3
        
        # Volatility indicators
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Volume indicators
        self.vwap_period = 20
        self.obv_period = 20
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        try:
            # Ensure we have enough data
            if len(df) < max(50, self.ema_slow_period, self.bb_period * 2):
                return None
            
            # 1. Trend Indicators
            df['ema_fast'] = EMAIndicator(close=df['close'], window=self.ema_fast_period).ema_indicator()
            df['ema_medium'] = EMAIndicator(close=df['close'], window=self.ema_medium_period).ema_indicator()
            df['ema_slow'] = EMAIndicator(close=df['close'], window=self.ema_slow_period).ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'], 
                       window_slow=self.macd_slow,
                       window_fast=self.macd_fast,
                       window_sign=self.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # 2. Momentum Indicators
            # RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df['high'], 
                                       low=df['low'], 
                                       close=df['close'],
                                       window=self.stoch_k,
                                       smooth_window=self.stoch_smooth,
                                       window_slow=self.stoch_d)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Awesome Oscillator
            df['ao'] = AwesomeOscillatorIndicator(high=df['high'], 
                                                low=df['low'], 
                                                window1=5, 
                                                window2=34).awesome_oscillator()
            
            # 3. Volatility Indicators
            # ATR
            df['atr'] = AverageTrueRange(high=df['high'], 
                                       low=df['low'], 
                                       close=df['close'], 
                                       window=self.atr_period).average_true_range()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], 
                              window=self.bb_period, 
                              window_dev=self.bb_std)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            
            # 4. Volume Indicators
            # Volume Weighted Average Price
            df['vwap'] = VolumeWeightedAveragePrice(high=df['high'],
                                                  low=df['low'],
                                                  close=df['close'],
                                                  volume=df['volume'],
                                                  window=self.vwap_period).volume_weighted_average_price()
            
            # On-Balance Volume
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], 
                                               volume=df['volume']).on_balance_volume()
            
            # 5. Price Action
            # Candlestick patterns (example: engulfing, doji, etc.)
            df['body'] = (df['close'] - df['open']).abs()
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_bullish'] = df['close'] > df['open']
            
            # Engulfing pattern
            prev = df.shift(1)
            df['bullish_engulfing'] = (
                (~prev['is_bullish']) & 
                (df['is_bullish']) &
                (df['close'] > prev['open']) & 
                (df['open'] < prev['close'])
            )
            
            df['bearish_engulfing'] = (
                (prev['is_bullish']) & 
                (~df['is_bullish']) &
                (df['open'] > prev['close']) & 
                (df['close'] < prev['open'])
            )
            
            # 6. Trend Strength
            adx = ADXIndicator(high=df['high'], 
                             low=df['low'], 
                             close=df['close'], 
                             window=14)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # 7. Ichimoku Cloud
            ichimoku = IchimokuIndicator(high=df['high'], 
                                       low=df['low'],
                                       window1=9,
                                       window2=26,
                                       window3=52)
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None
    
    async def analyze_and_trade(self):
        """Enhanced trading strategy with multiple confirmation signals."""
        try:
            if len(self.price_history) < 100:  # Need enough data for analysis
                return
                
            # Calculate indicators
            df = self.calculate_indicators(self.price_history)
            if df is None:
                return
                
            # Get the latest values
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Trading conditions
            long_conditions = [
                # Trend: Price above all EMAs and EMAs in correct order
                (current['close'] > current['ema_fast'] > current['ema_medium'] > current['ema_slow']),
                
                # Momentum: RSI not overbought
                (current['rsi'] < 70 and current['rsi'] > 50),
                
                # MACD: Bullish crossover
                (current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']),
                
                # Volume: Increasing volume on up moves
                (current['volume'] > df['volume'].rolling(20).mean().iloc[-1]),
                
                # Volatility: Price above VWAP
                (current['close'] > current['vwap']),
                
                # Trend strength: ADX > 25
                (current['adx'] > 25 and current['adx_pos'] > current['adx_neg']),
                
                # Price action: Bullish engulfing pattern
                (current['bullish_engulfing'])
            ]
            
            short_conditions = [
                # Trend: Price below all EMAs and EMAs in correct order
                (current['close'] < current['ema_fast'] < current['ema_medium'] < current['ema_slow']),
                
                # Momentum: RSI not oversold
                (current['rsi'] > 30 and current['rsi'] < 50),
                
                # MACD: Bearish crossover
                (current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']),
                
                # Volume: Increasing volume on down moves
                (current['volume'] > df['volume'].rolling(20).mean().iloc[-1]),
                
                # Volatility: Price below VWAP
                (current['close'] < current['vwap']),
                
                # Trend strength: ADX > 25
                (current['adx'] > 25 and current['adx_neg'] > current['adx_pos']),
                
                # Price action: Bearish engulfing pattern
                (current['bearish_engulfing'])
            ]
            
            # Calculate signal strength (0-100%)
            long_strength = sum(1 for condition in long_conditions if condition) / len(long_conditions)
            short_strength = sum(1 for condition in short_conditions if condition) / len(short_conditions)
            
            # Only trade if we have a strong signal (at least 70% of conditions met)
            min_confidence = 0.7
            
            if len(self.active_trades) < self.max_open_trades:
                # Long entry
                if long_strength >= min_confidence and long_strength > short_strength:
                    stop_loss = min(current['low'], current['bb_low'])
                    take_profit = current['close'] + (2 * (current['close'] - stop_loss))
                    await self.enter_trade('buy', stop_loss=stop_loss, take_profit=take_profit)
                
                # Short entry
                elif short_strength >= min_confidence and short_strength > long_strength:
                    stop_loss = max(current['high'], current['bb_high'])
                    take_profit = current['close'] - (2 * (stop_loss - current['close']))
                    await self.enter_trade('sell', stop_loss=stop_loss, take_profit=take_profit)
            
            # Manage open trades
            await self.manage_open_trades()
            
            # Log the analysis
            self.logger.info(f"Analysis - Long: {long_strength*100:.1f}% | Short: {short_strength*100:.1f}% | Price: {current['close']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced analyze_and_trade: {e}")

# Override the main function to use our enhanced scalper
async def main():
    scalper = EnhancedDivinaScalper()
    asyncio.create_task(scalper.connect_websocket())
    
    # Start the FastAPI server
    config = uvicorn.Config(
        app=scalper.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    scalper.logger.info(f"Starting {scalper.strategy_name}...")
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down enhanced Divina Scalper...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
