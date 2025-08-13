""
Signal generation and management for AthenaMyst:Divina.

This module handles the generation, validation, and management of trading signals.
"""
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import hashlib
import json

from loguru import logger

from ..models import Signal, SignalDirection, SignalStrength, Timeframe, Candle
from ..indicators import Indicators


class SignalManager:
    """Manages signal generation, validation, and deduplication."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the signal manager."""
        self.config = config or {}
        self.cooldown_minutes = self.config.get('cooldown_minutes', 15)
        self.recent_signals: Dict[str, datetime] = {}
        self.signal_history: List[Signal] = []
        
        # Signal strength thresholds
        self.strength_thresholds = {
            'strong': self.config.get('strong_threshold', 0.7),
            'moderate': self.config.get('moderate_threshold', 0.5),
            'weak': self.config.get('weak_threshold', 0.3)
        }
    
    def _generate_signal_id(self, signal: Signal) -> str:
        """Generate a unique ID for a signal."""
        # Create a unique hash based on signal properties
        signal_str = (
            f"{signal.pair}_{signal.timeframe}_"
            f"{signal.direction}_{signal.strength}_"
            f"{signal.timestamp.timestamp()}"
        )
        return hashlib.md5(signal_str.encode()).hexdigest()
    
    def _is_in_cooldown(self, pair: str, direction: SignalDirection) -> bool:
        """Check if a signal is in cooldown for the given pair and direction."""
        cooldown_key = f"{pair}_{direction}"
        last_signal_time = self.recent_signals.get(cooldown_key)
        
        if not last_signal_time:
            return False
            
        cooldown_end = last_signal_time + timedelta(minutes=self.cooldown_minutes)
        return datetime.utcnow() < cooldown_end
    
    def _is_duplicate_signal(self, signal: Signal, window_hours: int = 24) -> bool:
        """Check if a similar signal was recently generated."""
        if not self.signal_history:
            return False
            
        time_threshold = datetime.utcnow() - timedelta(hours=window_hours)
        
        for past_signal in reversed(self.signal_history):
            if past_signal.timestamp < time_threshold:
                break
                
            if (past_signal.pair == signal.pair and
                past_signal.direction == signal.direction and
                past_signal.timeframe == signal.timeframe):
                # Check if the signals are similar based on price and indicators
                price_diff = abs(past_signal.price - signal.price) / signal.price
                if price_diff < 0.01:  # Less than 1% price difference
                    return True
                    
                # Add more similarity checks here if needed
                
        return False
    
    def _calculate_signal_strength(
        self,
        indicators: Dict[str, Any],
        direction: SignalDirection
    ) -> SignalStrength:
        """Calculate signal strength based on indicators and market conditions."""
        # This is a simplified version - implement your own strength calculation logic
        score = 0.0
        
        # Example: Score based on RSI
        if 'rsi_14' in indicators and indicators['rsi_14']:
            rsi = indicators['rsi_14'][-1]
            if direction == SignalDirection.BUY and rsi < 40:
                score += 0.3
            elif direction == SignalDirection.SELL and rsi > 60:
                score += 0.3
        
        # Example: Score based on Ichimoku Cloud
        if 'ichimoku' in indicators:
            ichimoku = indicators['ichimoku']
            if (direction == SignalDirection.BUY and 
                'tenkan_sen' in ichimoku and 'kijun_sen' in ichimoku and
                ichimoku['tenkan_sen'][-1] > ichimoku['kijun_sen'][-1]):
                score += 0.4
            elif (direction == SignalDirection.SELL and 
                  'tenkan_sen' in ichimoku and 'kijun_sen' in ichimoku and
                  ichimoku['tenkan_sen'][-1] < ichimoku['kijun_sen'][-1]):
                score += 0.4
        
        # Example: Score based on volume
        if 'volume_profile' in indicators and 'volume' in indicators['volume_profile']:
            volume_ratio = indicators['volume_profile'].get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume
                score += 0.3
        
        # Determine strength based on score
        if score >= self.strength_thresholds['strong']:
            return SignalStrength.STRONG
        elif score >= self.strength_thresholds['moderate']:
            return SignalStrength.MODERATE
        elif score >= self.strength_thresholds['weak']:
            return SignalStrength.WEAK
        else:
            return SignalStrength.WEAK
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate a signal before processing."""
        # Check for required fields
        if not all([signal.pair, signal.timeframe, signal.direction, signal.price]):
            logger.warning(f"Invalid signal: missing required fields")
            return False
        
        # Check price validity
        if signal.price <= 0:
            logger.warning(f"Invalid signal price: {signal.price}")
            return False
        
        # Check timestamp (not in the future)
        if signal.timestamp > datetime.utcnow() + timedelta(minutes=5):
            logger.warning(f"Signal timestamp in the future: {signal.timestamp}")
            return False
        
        # Check cooldown
        if self._is_in_cooldown(signal.pair, signal.direction):
            logger.debug(f"Signal in cooldown: {signal.pair} {signal.direction}")
            return False
        
        # Check for duplicates
        if self._is_duplicate_signal(signal):
            logger.debug(f"Duplicate signal detected: {signal.pair} {signal.direction}")
            return False
        
        return True
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """Process and validate a new signal."""
        # Generate a unique ID for the signal
        signal.id = self._generate_signal_id(signal)
        
        # Validate the signal
        if not self.validate_signal(signal):
            return None
        
        # Calculate signal strength if not provided
        if not signal.strength and signal.indicators:
            signal.strength = self._calculate_signal_strength(
                signal.indicators,
                signal.direction
            )
        
        # Add timestamp if not provided
        if not signal.timestamp:
            signal.timestamp = datetime.utcnow()
        
        # Update cooldown
        cooldown_key = f"{signal.pair}_{signal.direction}"
        self.recent_signals[cooldown_key] = signal.timestamp
        
        # Add to history
        self.signal_history.append(signal)
        
        # Clean up old signals (keep last 1000)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        # Clean up old cooldown entries (older than 24 hours)
        expiration = datetime.utcnow() - timedelta(hours=24)
        self.recent_signals = {
            k: v for k, v in self.recent_signals.items()
            if v > expiration
        }
        
        logger.info(
            f"New signal: {signal.pair} {signal.timeframe} {signal.direction} "
            f"({signal.strength}) at {signal.price}"
        )
        
        return signal
    
    def get_recent_signals(
        self,
        pair: Optional[str] = None,
        direction: Optional[SignalDirection] = None,
        timeframe: Optional[Timeframe] = None,
        limit: int = 10
    ) -> List[Signal]:
        """Get recent signals matching the given filters."""
        signals = self.signal_history.copy()
        
        # Apply filters
        if pair is not None:
            signals = [s for s in signals if s.pair == pair]
        if direction is not None:
            signals = [s for s in signals if s.direction == direction]
        if timeframe is not None:
            signals = [s for s in signals if s.timeframe == timeframe]
        
        # Sort by timestamp (newest first)
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        
        return signals[:limit]
