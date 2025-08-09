#!/usr/bin/env python3
"""
Ichimoku Cloud Strategy for Crypto Trading

This module implements a multi-timeframe Ichimoku cloud strategy for scalping.
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

class IchimokuCloud:
    """Ichimoku Cloud technical analysis indicator."""
    
    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, displacement: int = 26):
        """
        Initialize the Ichimoku Cloud indicator.
        
        Args:
            tenkan: Period for Tenkan-sen (conversion line)
            kijun: Period for Kijun-sen (base line)
            senkou_b: Period for Senkou Span B (leading span B)
            displacement: Displacement for the cloud
        """
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.displacement = displacement
    
    def calculate(self, high: List[float], low: List[float], close: List[float]) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud indicators.
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of closing prices
            
        Returns:
            Dictionary containing all Ichimoku lines
        """
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = talib.MAX(high, timeperiod=self.tenkan)
        tenkan_low = talib.MIN(low, timeperiod=self.tenkan)
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = talib.MAX(high, timeperiod=self.kijun)
        kijun_low = talib.MIN(low, timeperiod=self.kijun)
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = talib.MAX(high, timeperiod=self.senkou_b)
        senkou_b_low = talib.MIN(low, timeperiod=self.senkou_b)
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2)
        
        # Chikou Span (Lagging Span)
        chikou_span = close
        
        # Displace the cloud forward
        senkou_span_a = np.roll(senkou_span_a, -self.displacement)
        senkou_span_b = np.roll(senkou_span_b, -self.displacement)
        
        # Fill NaN values with the last valid value
        senkou_span_a = pd.Series(senkou_span_a).fillna(method='ffill').values
        senkou_span_b = pd.Series(senkou_span_b).fillna(method='ffill').values
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }


def analyze_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26
) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """
    Analyze price action using Ichimoku Cloud.
    
    Args:
        df: DataFrame with OHLCV data
        tenkan: Tenkan-sen period
        kijun: Kijun-sen period
        senkou_b: Senkou Span B period
        displacement: Cloud displacement
        
    Returns:
        Tuple of (indicators, signals)
    """
    # Initialize Ichimoku Cloud
    ichimoku = IchimokuCloud(tenkan=tenkan, kijun=kijun, senkou_b=senkou_b, displacement=displacement)
    
    # Get OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate indicators
    indicators = ichimoku.calculate(high, low, close)
    
    # Generate signals
    signals = {
        'price_above_cloud': close[-1] > max(indicators['senkou_span_a'][-1], indicators['senkou_span_b'][-1]),
        'price_below_cloud': close[-1] < min(indicators['senkou_span_a'][-1], indicators['senkou_span_b'][-1]),
        'tenkan_above_kijun': indicators['tenkan_sen'][-1] > indicators['kijun_sen'][-1],
        'tenkan_below_kijun': indicators['tenkan_sen'][-1] < indicators['kijun_sen'][-1],
        'cloud_green': indicators['senkou_span_a'][-1] > indicators['senkou_span_b'][-1],
        'cloud_red': indicators['senkou_span_a'][-1] < indicators['senkou_span_b'][-1],
        'chikou_above_price': indicators['chikou_span'][-26] > close[-26] if len(close) > 26 else False,
        'chikou_below_price': indicators['chikou_span'][-26] < close[-26] if len(close) > 26 else False
    }
    
    return indicators, signals


def generate_trading_signals(
    ohlcv_data: Dict[str, pd.DataFrame],  # Dict of timeframes to DataFrames
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26
) -> Dict[str, bool]:
    """
    Generate trading signals based on multi-timeframe Ichimoku analysis.
    
    Args:
        ohlcv_data: Dictionary mapping timeframes to OHLCV DataFrames
        tenkan: Tenkan-sen period
        kijun: Kijun-sen period
        senkou_b: Senkou Span B period
        displacement: Cloud displacement
        
    Returns:
        Dictionary of trading signals
    """
    signals = {}
    
    # Analyze each timeframe
    for tf, df in ohlcv_data.items():
        if len(df) < max(tenkan, kijun, senkou_b) + displacement:
            logger.warning(f"Not enough data for {tf} timeframe. Need at least {max(tenkan, kijun, senkou_b) + displacement} candles.")
            continue
            
        _, tf_signals = analyze_ichimoku(df, tenkan, kijun, senkou_b, displacement)
        
        # Add timeframe prefix to signal names
        for signal_name, value in tf_signals.items():
            signals[f"{tf}_{signal_name}"] = value
    
    # Generate final signals based on all timeframes
    final_signals = {
        'strong_buy': all(signals.get(f"{tf}_price_above_cloud", False) and 
                         signals.get(f"{tf}_tenkan_above_kijun", False) and
                         signals.get(f"{tf}_cloud_green", False) and
                         signals.get(f"{tf}_chikou_above_price", False)
                         for tf in ohlcv_data.keys()),
        
        'buy': (sum(1 for tf in ohlcv_data.keys() 
                   if signals.get(f"{tf}_price_above_cloud", False) and
                      signals.get(f"{tf}_tenkan_above_kijun", False)) 
               >= len(ohlcv_data) * 0.7),  # At least 70% of timeframes agree
        
        'strong_sell': all(signals.get(f"{tf}_price_below_cloud", False) and 
                          signals.get(f"{tf}_tenkan_below_kijun", False) and
                          signals.get(f"{tf}_cloud_red", False) and
                          signals.get(f"{tf}_chikou_below_price", False)
                          for tf in ohlcv_data.keys()),
        
        'sell': (sum(1 for tf in ohlcv_data.keys()
                    if signals.get(f"{tf}_price_below_cloud", False) and
                       signals.get(f"{tf}_tenkan_below_kijun", False))
                >= len(ohlcv_data) * 0.7),  # At least 70% of timeframes agree
        
        'neutral': not (any(signals.get(f"{tf}_price_above_cloud", False) for tf in ohlcv_data.keys()) or
                       any(signals.get(f"{tf}_price_below_cloud", False) for tf in ohlcv_data.keys()))
    }
    
    # Add all individual signals to the final output
    final_signals.update(signals)
    
    return final_signals
