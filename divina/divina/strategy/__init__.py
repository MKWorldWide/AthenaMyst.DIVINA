""
Trading strategy implementation for AthenaMyst:Divina.

This module contains the core trading logic and signal generation.
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from ..models import Candle, Signal, SignalDirection, SignalStrength, Timeframe
from ..indicators import Indicators


class SignalType(str, Enum):
    """Types of trading signals."""
    ICHIMOKU = "ichimoku"
    RSI = "rsi"
    VWAP = "vwap"
    VOLUME = "volume"
    CONFIRMATION = "confirmation"


@dataclass
class SignalResult:
    """Result of a signal check."""
    direction: SignalDirection
    strength: SignalStrength
    indicators: Dict[str, Any]
    metadata: Dict[str, Any] = None


class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with configuration."""
        self.config = config or {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.signals: List[Signal] = []
    
    def check_ichimoku_signal(
        self, 
        candles: List[Candle],
        ichimoku_data: Dict[str, List[float]],
        current_price: float
    ) -> Optional[SignalResult]:
        """Check for Ichimoku Cloud signals."""
        if not ichimoku_data or len(ichimoku_data.get('tenkan_sen', [])) < 2:
            return None
        
        # Get the most recent values
        tenkan = ichimoku_data['tenkan_sen'][-1]
        kijun = ichimoku_data['kijun_sen'][-1]
        senkou_a = ichimoku_data['senkou_span_a'][-1]
        senkou_b = ichimoku_data['senkou_span_b'][-1]
        
        # Previous values for trend confirmation
        prev_tenkan = ichimoku_data['tenkan_sen'][-2] if len(ichimoku_data['tenkan_sen']) > 1 else tenkan
        prev_kijun = ichimoku_data['kijun_sen'][-2] if len(ichimoku_data['kijun_sen']) > 1 else kijun
        
        # Cloud calculations
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Signal conditions
        tenkan_above_kijun = tenkan > kijun
        price_above_cloud = current_price > cloud_top
        tenkan_rising = tenkan > prev_tenkan
        kijun_rising = kijun > prev_kijun
        
        # Determine signal strength
        strength = SignalStrength.MODERATE
        if tenkan_rising and kijun_rising and tenkan_above_kijun and price_above_cloud:
            strength = SignalStrength.STRONG
        
        # Generate signals
        if tenkan_above_kijun and price_above_cloud:
            return SignalResult(
                direction=SignalDirection.BUY,
                strength=strength,
                indicators={
                    'tenkan_sen': tenkan,
                    'kijun_sen': kijun,
                    'senkou_span_a': senkou_a,
                    'senkou_span_b': senkou_b,
                    'cloud_top': cloud_top,
                    'cloud_bottom': cloud_bottom
                }
            )
        elif not tenkan_above_kijun and not price_above_cloud:
            return SignalResult(
                direction=SignalDirection.SELL,
                strength=strength,
                indicators={
                    'tenkan_sen': tenkan,
                    'kijun_sen': kijun,
                    'senkou_span_a': senkou_a,
                    'senkou_span_b': senkou_b,
                    'cloud_top': cloud_top,
                    'cloud_bottom': cloud_bottom
                }
            )
        
        return None
    
    def check_rsi_signal(
        self, 
        candles: List[Candle], 
        rsi_values: List[float],
        overbought: float = 70.0,
        oversold: float = 30.0
    ) -> Optional[SignalResult]:
        """Check for RSI signals."""
        if not rsi_values or len(rsi_values) < 2:
            return None
        
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else current_rsi
        
        # Check for overbought/oversold conditions
        if current_rsi > overbought and prev_rsi <= overbought:
            return SignalResult(
                direction=SignalDirection.SELL,
                strength=SignalStrength.MODERATE,
                indicators={'rsi': current_rsi}
            )
        elif current_rsi < oversold and prev_rsi >= oversold:
            return SignalResult(
                direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE,
                indicators={'rsi': current_rsi}
            )
        
        # Check for RSI divergence (simplified)
        if len(rsi_values) > 10:
            price_trend = np.polyfit(
                range(len(candles[-10:])), 
                [c.close for c in candles[-10:]], 
                1
            )[0]
            rsi_trend = np.polyfit(range(len(rsi_values[-10:])), rsi_values[-10:], 1)[0]
            
            if price_trend > 0 and rsi_trend < 0:  # Bearish divergence
                return SignalResult(
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.WEAK,
                    indicators={'rsi': current_rsi, 'divergence': 'bearish'}
                )
            elif price_trend < 0 and rsi_trend > 0:  # Bullish divergence
                return SignalResult(
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.WEAK,
                    indicators={'rsi': current_rsi, 'divergence': 'bullish'}
                )
        
        return None
    
    def check_vwap_signal(
        self, 
        candles: List[Candle], 
        vwap_values: List[float]
    ) -> Optional[SignalResult]:
        """Check for VWAP signals."""
        if not vwap_values or len(candles) != len(vwap_values):
            return None
        
        current_candle = candles[-1]
        current_vwap = vwap_values[-1]
        prev_vwap = vwap_values[-2] if len(vwap_values) > 1 else current_vwap
        
        # Check if price crossed VWAP
        if current_candle.close > current_vwap and current_candle.close > prev_vwap:
            return SignalResult(
                direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE,
                indicators={'vwap': current_vwap}
            )
        elif current_candle.close < current_vwap and current_candle.close < prev_vwap:
            return SignalResult(
                direction=SignalDirection.SELL,
                strength=SignalStrength.MODERATE,
                indicators={'vwap': current_vwap}
            )
        
        return None
    
    def check_volume_signal(
        self, 
        candles: List[Candle], 
        volume_ma_period: int = 20,
        volume_multiplier: float = 1.5
    ) -> Optional[SignalResult]:
        """Check for volume surge signals."""
        if not candles or len(candles) < volume_ma_period + 1:
            return None
        
        # Ensure we have volume data
        if not all(hasattr(c, 'volume') and c.volume is not None for c in candles):
            return None
        
        volumes = [c.volume for c in candles]  # type: ignore
        current_volume = volumes[-1]
        
        # Calculate volume moving average
        volume_ma = np.mean(volumes[-volume_ma_period-1:-1])
        
        # Check for volume surge
        if current_volume > volume_ma * volume_multiplier:
            # Check if volume is supporting price movement
            price_change = (candles[-1].close - candles[-2].close) / candles[-2].close
            
            if price_change > 0:  # Bullish volume surge
                return SignalResult(
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.STRONG,
                    indicators={
                        'volume': current_volume,
                        'volume_ma': volume_ma,
                        'volume_ratio': current_volume / volume_ma if volume_ma > 0 else 0
                    }
                )
            elif price_change < 0:  # Bearish volume surge
                return SignalResult(
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.STRONG,
                    indicators={
                        'volume': current_volume,
                        'volume_ma': volume_ma,
                        'volume_ratio': current_volume / volume_ma if volume_ma > 0 else 0
                    }
                )
        
        return None
    
    def check_confirmation_signal(
        self,
        signal: SignalResult,
        higher_tf_candles: List[Candle],
        current_price: float
    ) -> Optional[SignalResult]:
        """Check if signal is confirmed by higher timeframe."""
        if not higher_tf_candles:
            return signal
        
        # Simple trend confirmation using higher timeframe
        higher_tf_trend = self._get_trend(higher_tf_candles)
        
        # Only confirm if higher timeframe trend aligns with signal
        if (higher_tf_trend > 0 and signal.direction == SignalDirection.BUY) or \
           (higher_tf_trend < 0 and signal.direction == SignalDirection.SELL):
            # Increase signal strength if confirmed
            if signal.strength == SignalStrength.WEAK:
                signal.strength = SignalStrength.MODERATE
            elif signal.strength == SignalStrength.MODERATE:
                signal.strength = SignalStrength.STRONG
            
            signal.metadata = signal.metadata or {}
            signal.metadata['higher_tf_confirmed'] = True
            
            return signal
        
        # If not confirmed, weaken the signal
        if signal.strength == SignalStrength.STRONG:
            signal.strength = SignalStrength.MODERATE
        elif signal.strength == SignalStrength.MODERATE:
            signal.strength = SignalStrength.WEAK
        
        signal.metadata = signal.metadata or {}
        signal.metadata['higher_tf_confirmed'] = False
        
        return signal
    
    def _get_trend(self, candles: List[Candle], period: int = 20) -> float:
        """Calculate trend direction (-1 to 1) using linear regression."""
        if len(candles) < period:
            return 0.0
        
        closes = [c.close for c in candles[-period:]]
        x = np.arange(len(closes))
        
        # Calculate linear regression
        slope, _ = np.polyfit(x, closes, 1)
        
        # Normalize slope to -1 to 1 range
        max_slope = np.max(np.abs(np.diff(closes))) * len(closes)
        if max_slope == 0:
            return 0.0
            
        return float(np.clip(slope / max_slope, -1, 1))
    
    def generate_signals(
        self,
        pair: str,
        timeframe: Timeframe,
        candles: List[Candle],
        higher_tf_candles: Optional[List[Candle]] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """Generate trading signals based on the given candles and indicators."""
        signals = []
        current_time = datetime.utcnow()
        current_price = candles[-1].close if candles else 0
        
        # Check cooldown period
        last_signal_time = self.last_signal_time.get(pair)
        if last_signal_time and (current_time - last_signal_time).total_seconds() < 3600:  # 1h cooldown
            return signals
        
        # Calculate indicators if not provided
        if indicators is None:
            indicators = Indicators.calculate_all(candles)
        
        # Check each signal type
        signal_results = {}
        
        # Ichimoku signal
        if 'ichimoku' in indicators:
            ichimoku_signal = self.check_ichimoku_signal(
                candles, 
                indicators['ichimoku'],
                current_price
            )
            if ichimoku_signal:
                signal_results[SignalType.ICHIMOKU] = ichimoku_signal
        
        # RSI signal
        if 'rsi_14' in indicators:
            rsi_signal = self.check_rsi_signal(candles, indicators['rsi_14'])
            if rsi_signal:
                signal_results[SignalType.RSI] = rsi_signal
        
        # VWAP signal
        if 'vwap' in indicators and all(hasattr(c, 'volume') for c in candles):
            vwap_signal = self.check_vwap_signal(candles, indicators['vwap'])
            if vwap_signal:
                signal_results[SignalType.VWAP] = vwap_signal
        
        # Volume signal
        if all(hasattr(c, 'volume') for c in candles):
            volume_signal = self.check_volume_signal(
                candles,
                volume_multiplier=self.config.get('volume_multiplier', 1.5)
            )
            if volume_signal:
                signal_results[SignalType.VOLUME] = volume_signal
        
        # Check for confirmation on higher timeframe
        if higher_tf_candles and signal_results:
            for signal_type, signal in signal_results.items():
                confirmed_signal = self.check_confirmation_signal(
                    signal,
                    higher_tf_candles,
                    current_price
                )
                if confirmed_signal:
                    signal_results[signal_type] = confirmed_signal
        
        # Generate final signals
        for signal_type, result in signal_results.items():
            # Only take strong signals or multiple confirming signals
            if result.strength == SignalStrength.STRONG or len(signal_results) > 1:
                signal = Signal(
                    pair=pair,
                    timeframe=timeframe,
                    direction=result.direction,
                    strength=result.strength,
                    price=current_price,
                    indicators=result.indicators,
                    metadata=result.metadata or {}
                )
                signals.append(signal)
                
                # Update last signal time
                self.last_signal_time[pair] = current_time
        
        # Store signals
        self.signals.extend(signals)
        return signals
    
    def get_stop_loss_take_profit(
        self,
        signal: Signal,
        atr: Optional[float] = None,
        risk_reward_ratio: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate stop loss and take profit levels."""
        price = signal.price
        
        # Default to 1% stop loss if ATR is not available
        if atr is None:
            atr = price * 0.01
        
        if signal.direction == SignalDirection.BUY:
            stop_loss = price - (atr * 1.5)  # 1.5x ATR for stop loss
            take_profit = price + (atr * 1.5 * risk_reward_ratio)
        else:  # SELL
            stop_loss = price + (atr * 1.5)  # 1.5x ATR for stop loss
            take_profit = price - (atr * 1.5 * risk_reward_ratio)
        
        # Calculate risk percentage
        risk_pct = abs((stop_loss - price) / price) * 100
        
        return stop_loss, take_profit, risk_pct
