""
Unit tests for trading rules and signal generation.

This module contains tests for the trading rules, including buy/sell signals,
deduplication, and risk management logic.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from divina.strategy.rules import (
    generate_signals,
    Signal,
    SignalType,
    SignalDirection,
    SignalConfidence,
    deduplicate_signals,
    filter_signals_by_priority,
    apply_cooldown_period,
    calculate_position_size,
    calculate_risk_reward_ratio,
)
from divina.data.manager import DataManager
from divina.indicators import calculate_ichimoku, calculate_rsi

# Directory containing test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Test data file paths
SIGNAL_TEST_DATA = TEST_DATA_DIR / "signal_test_data.parquet"

# Test configuration
TEST_CONFIG = {
    "symbols": ["EURUSD"],
    "timeframes": ["1h", "4h"],
    "signal_timeframe": "1h",
    "confirmation_timeframe": "4h",
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou_span_b": 52,
    "min_confidence": 0.7,
    "cooldown_period": "4h",
    "risk_per_trade": 0.01,
    "account_balance": 10000.0,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
}


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, pd.DataFrame]:
    """Create sample OHLCV data for multiple timeframes."""
    np.random.seed(42)
    base_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    n_bars = 100
    
    # Generate base 1h data
    dates_1h = [base_date + timedelta(hours=i) for i in range(n_bars)]
    close_1h = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0005)
    
    # Create OHLCV data with some structure
    data_1h = {
        'open': close_1h - np.abs(np.random.randn(n_bars) * 0.0005),
        'high': close_1h + np.abs(np.random.randn(n_bars) * 0.0005),
        'low': close_1h - np.abs(np.random.randn(n_bars) * 0.0005),
        'close': close_1h,
        'volume': (np.random.rand(n_bars) * 1000).astype(int) + 100,
    }
    
    df_1h = pd.DataFrame(data_1h, index=dates_1h)
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return {
        '1h': df_1h,
        '4h': df_4h,
    }


@pytest.fixture
def sample_signals() -> List[Signal]:
    """Create sample signals for testing."""
    base_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    return [
        Signal(
            symbol="EURUSD",
            signal_type=SignalType.ICHIMOKU,
            direction=SignalDirection.LONG,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1150,
            timeframe="1h",
            timestamp=base_time,
            confidence=0.85,
            indicators={
                'tenkan_sen': 1.0990,
                'kijun_sen': 1.0980,
                'senkou_span_a': 1.0970,
                'senkou_span_b': 1.0960,
                'rsi': 45.0,
            },
        ),
        Signal(
            symbol="EURUSD",
            signal_type=SignalType.RSI_DIVERGENCE,
            direction=SignalDirection.SHORT,
            entry_price=1.1050,
            stop_loss=1.1100,
            take_profit=1.0850,
            timeframe="4h",
            timestamp=base_time + timedelta(hours=1),
            confidence=0.75,
            indicators={
                'rsi': 72.0,
                'divergence': 'bearish',
            },
        ),
        Signal(
            symbol="EURUSD",
            signal_type=SignalType.ICHIMOKU,
            direction=SignalDirection.LONG,
            entry_price=1.1020,
            stop_loss=1.0970,
            take_profit=1.1220,
            timeframe="1h",
            timestamp=base_time + timedelta(hours=2),
            confidence=0.92,
            indicators={
                'tenkan_sen': 1.1010,
                'kijun_sen': 1.1000,
                'senkou_span_a': 1.0990,
                'senkou_span_b': 1.0980,
                'rsi': 48.0,
            },
        ),
    ]


def test_generate_signals(sample_ohlcv_data):
    """Test signal generation with sample data."""
    # Prepare data manager with sample data
    data_manager = DataManager()
    for tf, df in sample_ohlcv_data.items():
        data_manager.update_market_data("EURUSD", tf, df)
    
    # Generate signals
    signals = generate_signals(
        data_manager=data_manager,
        symbol="EURUSD",
        signal_timeframe="1h",
        confirmation_timeframe="4h",
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        ichimoku_tenkan=9,
        ichimoku_kijun=26,
        ichimoku_senkou_span_b=52,
    )
    
    # Basic validation
    assert isinstance(signals, list)
    
    # Check signal properties
    for signal in signals:
        assert isinstance(signal, Signal)
        assert signal.symbol == "EURUSD"
        assert signal.timeframe in ["1h", "4h"]
        assert signal.direction in [SignalDirection.LONG, SignalDirection.SHORT]
        assert signal.confidence >= 0.0 and signal.confidence <= 1.0
        assert signal.entry_price > 0
        assert signal.stop_loss > 0
        assert signal.take_profit > 0
        assert signal.timestamp.tzinfo == timezone.utc
        
        # Validate indicators
        assert isinstance(signal.indicators, dict)
        
        # Check specific indicator values based on signal type
        if signal.signal_type == SignalType.ICHIMOKU:
            assert 'tenkan_sen' in signal.indicators
            assert 'kijun_sen' in signal.indicators
            assert 'senkou_span_a' in signal.indicators
            assert 'senkou_span_b' in signal.indicators
        elif signal.signal_type == SignalType.RSI_DIVERGENCE:
            assert 'rsi' in signal.indicators
            assert 'divergence' in signal.indicators


def test_deduplicate_signals(sample_signals):
    """Test signal deduplication logic."""
    # Add a duplicate signal (same type, direction, similar price and time)
    duplicate_signal = Signal(
        symbol="EURUSD",
        signal_type=SignalType.ICHIMOKU,
        direction=SignalDirection.LONG,
        entry_price=1.1005,  # Slightly different price
        stop_loss=1.0950,
        take_profit=1.1150,
        timeframe="1h",
        timestamp=sample_signals[0].timestamp + timedelta(minutes=15),  # Close in time
        confidence=0.82,  # Slightly different confidence
        indicators={
            'tenkan_sen': 1.0990,
            'kijun_sen': 1.0980,
            'senkou_span_a': 1.0970,
            'senkou_span_b': 1.0960,
            'rsi': 46.0,
        },
    )
    
    signals_with_duplicate = sample_signals + [duplicate_signal]
    
    # Deduplicate signals
    deduped = deduplicate_signals(signals_with_duplicate)
    
    # Should keep the higher confidence signal
    assert len(deduped) == len(sample_signals)  # One signal should be removed
    
    # The remaining signal should be the higher confidence one
    for signal in deduped:
        if (
            signal.signal_type == SignalType.ICHIMOKU and 
            signal.direction == SignalDirection.LONG and
            signal.timestamp.date() == sample_signals[0].timestamp.date()
        ):
            assert signal.confidence == max(sample_signals[0].confidence, duplicate_signal.confidence)


def test_filter_signals_by_priority(sample_signals):
    """Test filtering signals by priority and confidence."""
    # Filter signals with minimum confidence
    filtered = filter_signals_by_priority(
        sample_signals,
        min_confidence=0.8,
        priority_order=[SignalType.ICHIMOKU, SignalType.RSI_DIVERGENCE]
    )
    
    # Should filter out the RSI signal with confidence 0.75
    assert len(filtered) == 2
    assert all(s.confidence >= 0.8 for s in filtered)
    assert all(s.signal_type == SignalType.ICHIMOKU for s in filtered)
    
    # Test with different priority order
    filtered = filter_signals_by_priority(
        sample_signals,
        min_confidence=0.7,
        priority_order=[SignalType.RSI_DIVERGENCE, SignalType.ICHIMOKU]
    )
    
    # Should include all signals, but RSI first
    assert len(filtered) == 3
    assert filtered[0].signal_type == SignalType.RSI_DIVERGENCE


def test_apply_cooldown_period(sample_signals):
    """Test cooldown period application."""
    # Add a signal that's too close in time to the first one
    new_signal = Signal(
        symbol="EURUSD",
        signal_type=SignalType.ICHIMOKU,
        direction=SignalDirection.SHORT,  # Different direction
        entry_price=1.0980,
        stop_loss=1.1030,
        take_profit=1.0880,
        timeframe="1h",
        timestamp=sample_signals[0].timestamp + timedelta(minutes=30),  # Within cooldown
        confidence=0.88,
        indicators={},
    )
    
    signals_with_cooldown = sample_signals + [new_signal]
    
    # Apply cooldown period (2 hours)
    filtered = apply_cooldown_period(
        signals_with_cooldown,
        cooldown_period=timedelta(hours=2)
    )
    
    # Should keep the first signal and the new one (different direction)
    # and filter out the second signal of the same type
    assert len(filtered) == 3  # Original 3 signals (one was a duplicate)
    
    # Add another signal of the same type and direction within cooldown
    another_signal = Signal(
        symbol="EURUSD",
        signal_type=SignalType.ICHIMOKU,
        direction=SignalDirection.LONG,  # Same direction as first
        entry_price=1.1010,
        stop_loss=1.0960,
        take_profit=1.1160,
        timeframe="1h",
        timestamp=sample_signals[0].timestamp + timedelta(minutes=45),  # Within cooldown
        confidence=0.90,
        indicators={},
    )
    
    signals_with_cooldown.append(another_signal)
    
    # Apply cooldown period again
    filtered = apply_cooldown_period(
        signals_with_cooldown,
        cooldown_period=timedelta(hours=2)
    )
    
    # Should keep the higher confidence signal of the same type/direction
    assert len(filtered) == 3  # Still 3 signals
    long_signals = [s for s in filtered if s.direction == SignalDirection.LONG]
    assert len(long_signals) == 1
    assert long_signals[0].confidence == max(
        s.confidence for s in signals_with_cooldown 
        if s.signal_type == SignalType.ICHIMOKU and s.direction == SignalDirection.LONG
    )


def test_calculate_position_size():
    """Test position size calculation."""
    # Test with default risk (1% of account)
    account_balance = 10000.0
    risk_per_trade = 0.01  # 1%
    entry_price = 1.1000
    stop_loss = 1.0950
    
    # Calculate position size
    position_size = calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_balance=account_balance,
        risk_per_trade=risk_per_trade,
    )
    
    # Expected risk amount: 10000 * 0.01 = 100
    # Risk per unit: (1.1000 - 1.0950) = 0.0050
    # Position size: 100 / 0.0050 = 20000
    expected_size = (account_balance * risk_per_trade) / (entry_price - stop_loss)
    assert abs(position_size - expected_size) < 0.01
    
    # Test with zero risk (should return 0)
    position_size = calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_balance=account_balance,
        risk_per_trade=0.0,
    )
    assert position_size == 0.0
    
    # Test with stop loss equal to entry (should raise ValueError)
    with pytest.raises(ValueError):
        calculate_position_size(
            entry_price=entry_price,
            stop_loss=entry_price,  # Same as entry
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
        )


def test_calculate_risk_reward_ratio():
    """Test risk/reward ratio calculation."""
    # Test with valid values
    entry = 1.1000
    stop_loss = 1.0950
    take_profit = 1.1150
    
    # Calculate risk/reward
    rr_ratio = calculate_risk_reward_ratio(entry, stop_loss, take_profit)
    
    # Risk: 1.1000 - 1.0950 = 0.0050
    # Reward: 1.1150 - 1.1000 = 0.0150
    # Ratio: 0.0150 / 0.0050 = 3.0
    expected_ratio = (take_profit - entry) / (entry - stop_loss)
    assert abs(rr_ratio - expected_ratio) < 0.0001
    
    # Test with zero risk (should return infinity)
    rr_ratio = calculate_risk_reward_ratio(entry, entry, take_profit)
    assert rr_ratio == float('inf')
    
    # Test with zero reward (should return 0)
    rr_ratio = calculate_risk_reward_ratio(entry, stop_loss, entry)
    assert rr_ratio == 0.0


def test_signal_serialization():
    """Test signal serialization and deserialization."""
    # Create a signal
    signal = Signal(
        symbol="EURUSD",
        signal_type=SignalType.ICHIMOKU,
        direction=SignalDirection.LONG,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1150,
        timeframe="1h",
        timestamp=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
        confidence=0.85,
        indicators={
            'tenkan_sen': 1.0990,
            'kijun_sen': 1.0980,
            'rsi': 45.0,
        },
    )
    
    # Convert to dict
    signal_dict = signal.dict()
    
    # Convert back to Signal
    new_signal = Signal(**signal_dict)
    
    # Should be equal
    assert signal == new_signal
    
    # Test JSON serialization
    signal_json = signal.json()
    new_signal_from_json = Signal.parse_raw(signal_json)
    assert signal == new_signal_from_json
