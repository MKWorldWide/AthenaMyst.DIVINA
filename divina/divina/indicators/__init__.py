""
Technical indicators for AthenaMyst:Divina.

This module provides various technical indicators used for market analysis
and trading signal generation.
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from ..models import Candle, IndicatorValues, Timeframe


class Indicators:
    """Container for all technical indicators."""
    
    @staticmethod
    def sma(candles: List[Candle], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        if not candles or len(candles) < period:
            return []
            
        closes = [c.close for c in candles]
        return list(pd.Series(closes).rolling(window=period).mean().dropna())
    
    @staticmethod
    def ema(candles: List[Candle], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if not candles or len(candles) < period:
            return []
            
        closes = [c.close for c in candles]
        return list(pd.Series(closes).ewm(span=period, adjust=False).mean())
    
    @staticmethod
    def rsi(candles: List[Candle], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if not candles or len(candles) < period + 1:
            return []
            
        df = pd.DataFrame([{
            'close': c.close,
            'timestamp': c.timestamp
        } for c in candles])
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.dropna().tolist()
    
    @staticmethod
    def ichimoku(
        candles: List[Candle], 
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> Dict[str, List[float]]:
        """Calculate Ichimoku Cloud indicators."""
        if not candles or len(candles) < max(tenkan_period, kijun_period, senkou_b_period) + displacement:
            return {}
            
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'timestamp': c.timestamp
        } for c in candles])
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_low = df['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen.dropna().tolist(),
            'kijun_sen': kijun_sen.dropna().tolist(),
            'senkou_span_a': senkou_span_a.dropna().tolist(),
            'senkou_span_b': senkou_span_b.dropna().tolist(),
            'chikou_span': chikou_span.dropna().tolist()
        }
    
    @staticmethod
    def vwap(candles: List[Candle]) -> List[float]:
        """Calculate Volume Weighted Average Price."""
        if not candles or not all(hasattr(c, 'volume') and c.volume is not None for c in candles):
            return []
            
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'timestamp': c.timestamp
        } for c in candles])
        
        # Typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Cumulative volume * typical price / cumulative volume
        vwap = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return vwap.dropna().tolist()
    
    @staticmethod
    def volume_profile(candles: List[Candle], bins: int = 20) -> Dict[str, Any]:
        """Calculate volume profile."""
        if not candles or not all(hasattr(c, 'volume') and c.volume is not None for c in candles):
            return {}
            
        prices = [c.close for c in candles]
        volumes = [c.volume for c in candles]  # type: ignore
        
        # Create price bins
        min_price, max_price = min(prices), max(prices)
        price_range = max_price - min_price
        bin_size = price_range / bins
        
        # Initialize volume at price levels
        volume_at_price = {}
        for i in range(len(prices)):
            price = prices[i]
            volume = volumes[i]
            bin_idx = int((price - min_price) / bin_size)
            bin_price = min_price + (bin_idx * bin_size)
            
            if bin_price in volume_at_price:
                volume_at_price[bin_price] += volume
            else:
                volume_at_price[bin_price] = volume
        
        # Convert to lists for plotting
        price_levels = sorted(volume_at_price.keys())
        volume_levels = [volume_at_price[p] for p in price_levels]
        
        return {
            'price_levels': price_levels,
            'volume_levels': volume_levels,
            'poc': max(volume_at_price, key=volume_at_price.get),  # Point of Control
            'vah': max(volume_at_price.keys()),  # Value Area High
            'val': min(volume_at_price.keys())   # Value Area Low
        }
    
    @staticmethod
    def atr(candles: List[Candle], period: int = 14) -> List[float]:
        """Calculate Average True Range."""
        if not candles or len(candles) < period + 1:
            return []
            
        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'timestamp': c.timestamp
        } for c in candles])
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr = df['tr'].rolling(window=period).mean()
        
        return atr.dropna().tolist()
    
    @staticmethod
    def macd(
        candles: List[Candle], 
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, List[float]]:
        """Calculate MACD indicator."""
        if not candles or len(candles) < slow_period + signal_period:
            return {}
            
        closes = [c.close for c in candles]
        
        # Calculate EMAs
        ema_fast = pd.Series(closes).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(closes).ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD and signal line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.dropna().tolist(),
            'signal': signal.dropna().tolist(),
            'histogram': histogram.dropna().tolist()
        }
    
    @classmethod
    def calculate_all(
        cls,
        candles: List[Candle],
        indicators_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate all configured indicators."""
        if indicators_config is None:
            indicators_config = {
                'sma': [20, 50, 200],
                'ema': [9, 21, 50, 200],
                'rsi': [14],
                'ichimoku': {},
                'vwap': {},
                'atr': [14],
                'macd': {},
            }
        
        results = {}
        
        # Calculate each indicator
        if 'sma' in indicators_config:
            for period in indicators_config['sma']:
                results[f'sma_{period}'] = cls.sma(candles, period)
        
        if 'ema' in indicators_config:
            for period in indicators_config['ema']:
                results[f'ema_{period}'] = cls.ema(candles, period)
        
        if 'rsi' in indicators_config:
            for period in indicators_config['rsi']:
                results[f'rsi_{period}'] = cls.rsi(candles, period)
        
        if 'ichimoku' in indicators_config:
            results['ichimoku'] = cls.ichimoku(candles, **indicators_config.get('ichimoku', {}))
        
        if 'vwap' in indicators_config and all(hasattr(c, 'volume') for c in candles):
            results['vwap'] = cls.vwap(candles)
        
        if 'atr' in indicators_config:
            for period in indicators_config['atr']:
                results[f'atr_{period}'] = cls.atr(candles, period)
        
        if 'macd' in indicators_config:
            results['macd'] = cls.macd(candles, **indicators_config.get('macd', {}))
        
        return results
