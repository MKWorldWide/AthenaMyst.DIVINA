""
Unit tests for technical indicators.

This module contains golden data tests for all technical indicators used in AthenaMyst:Divina.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from divina.indicators import (
    calculate_ichimoku,
    calculate_rsi,
    calculate_vwap,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_stochastic,
    calculate_adx,
    calculate_obv,
    calculate_supertrend,
)

# Directory containing test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Test data file paths
ICHIMOKU_TEST_DATA = TEST_DATA_DIR / "ichimoku_test_data.csv"
RSI_TEST_DATA = TEST_DATA_DIR / "rsi_test_data.csv"
VWAP_TEST_DATA = TEST_DATA_DIR / "vwap_test_data.csv"
ATR_TEST_DATA = TEST_DATA_DIR / "atr_test_data.csv"
BBANDS_TEST_DATA = TEST_DATA_DIR / "bbands_test_data.csv"
MACD_TEST_DATA = TEST_DATA_DIR / "macd_test_data.csv"
STOCHASTIC_TEST_DATA = TEST_DATA_DIR / "stochastic_test_data.csv"
ADX_TEST_DATA = TEST_DATA_DIR / "adx_test_data.csv"
OBV_TEST_DATA = TEST_DATA_DIR / "obv_test_data.csv"
SUPERTREND_TEST_DATA = TEST_DATA_DIR / "supertrend_test_data.csv"

# Helper function to load test data
def load_test_data(file_path: Path) -> pd.DataFrame:
    """Load test data from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    df = pd.read_csv(
        file_path,
        parse_dates=['date'],
        date_parser=lambda x: pd.to_datetime(x, utc=True)
    )
    df.set_index('date', inplace=True)
    return df

# Fixture to create a sample OHLCV DataFrame
@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D', tz='UTC')
    n = len(date_rng)
    
    # Generate random walk for close prices
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    
    # Generate other OHLCV data
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_prices = close.shift(1).fillna(close[0] - np.abs(np.random.randn()))
    volume = (np.random.rand(n) * 1000).astype(int)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_rng)

# Test Ichimoku Cloud
def test_ichimoku_calculation():
    """Test Ichimoku Cloud calculation against golden data."""
    # Load test data
    test_data = load_test_data(ICHIMOKU_TEST_DATA)
    
    # Calculate Ichimoku
    result = calculate_ichimoku(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        chikou_shift=26,
        senkou_shift=26
    )
    
    # Compare with expected values (with some tolerance for floating point)
    expected_columns = [
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 
        'senkou_span_b', 'chikou_span'
    ]
    
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
        
        # Skip NaNs in comparison
        mask = ~result[col].isna() & ~test_data[col].isna()
        
        # Check if any non-NaN values to compare
        if mask.any():
            assert np.allclose(
                result[col][mask].astype(float),
                test_data[col][mask].astype(float),
                rtol=1e-3,
                atol=1e-4,
                equal_nan=True
            ), f"Mismatch in {col}"

# Test RSI
def test_rsi_calculation():
    """Test RSI calculation against golden data."""
    # Load test data
    test_data = load_test_data(RSI_TEST_DATA)
    
    # Calculate RSI
    result = calculate_rsi(test_data['close'], period=14)
    
    # Compare with expected values
    assert_series_equal(
        result.round(4),
        test_data['rsi'].round(4),
        check_names=False,
        check_exact=False,
        rtol=1e-3,
        atol=1e-4
    )

# Test VWAP
def test_vwap_calculation():
    """Test VWAP calculation against golden data."""
    # Load test data
    test_data = load_test_data(VWAP_TEST_DATA)
    
    # Calculate VWAP
    result = calculate_vwap(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        volume=test_data['volume']
    )
    
    # Compare with expected values
    assert_series_equal(
        result.round(4),
        test_data['vwap'].round(4),
        check_names=False,
        check_exact=False,
        rtol=1e-3,
        atol=1e-4
    )

# Test ATR
def test_atr_calculation():
    """Test ATR calculation against golden data."""
    # Load test data
    test_data = load_test_data(ATR_TEST_DATA)
    
    # Calculate ATR
    result = calculate_atr(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        period=14
    )
    
    # Compare with expected values
    assert_series_equal(
        result.round(4),
        test_data['atr'].round(4),
        check_names=False,
        check_exact=False,
        rtol=1e-3,
        atol=1e-4
    )

# Test Bollinger Bands
def test_bollinger_bands_calculation():
    """Test Bollinger Bands calculation against golden data."""
    # Load test data
    test_data = load_test_data(BBANDS_TEST_DATA)
    
    # Calculate Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        close=test_data['close'],
        period=20,
        num_std=2.0
    )
    
    # Compare with expected values
    for actual, expected_col in zip(
        [upper, middle, lower],
        ['bb_upper', 'bb_middle', 'bb_lower']
    ):
        assert_series_equal(
            actual.round(4),
            test_data[expected_col].round(4),
            check_names=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-4
        )

# Test MACD
def test_macd_calculation():
    """Test MACD calculation against golden data."""
    # Load test data
    test_data = load_test_data(MACD_TEST_DATA)
    
    # Calculate MACD
    macd, signal, hist = calculate_macd(
        close=test_data['close'],
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    # Compare with expected values
    for actual, expected_col in zip(
        [macd, signal, hist],
        ['macd', 'macd_signal', 'macd_hist']
    ):
        assert_series_equal(
            actual.round(4),
            test_data[expected_col].round(4),
            check_names=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-4
        )

# Test Stochastic Oscillator
def test_stochastic_calculation():
    """Test Stochastic Oscillator calculation against golden data."""
    # Load test data
    test_data = load_test_data(STOCHASTIC_TEST_DATA)
    
    # Calculate Stochastic
    k, d = calculate_stochastic(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        k_period=14,
        d_period=3,
        smooth_k=3
    )
    
    # Compare with expected values
    for actual, expected_col in zip(
        [k, d],
        ['stoch_k', 'stoch_d']
    ):
        assert_series_equal(
            actual.round(4),
            test_data[expected_col].round(4),
            check_names=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-4
        )

# Test ADX
def test_adx_calculation():
    """Test ADX calculation against golden data."""
    # Load test data
    test_data = load_test_data(ADX_TEST_DATA)
    
    # Calculate ADX
    adx, plus_di, minus_di = calculate_adx(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        period=14
    )
    
    # Compare with expected values
    for actual, expected_col in zip(
        [adx, plus_di, minus_di],
        ['adx', 'plus_di', 'minus_di']
    ):
        assert_series_equal(
            actual.round(4),
            test_data[expected_col].round(4),
            check_names=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-4
        )

# Test OBV
def test_obv_calculation():
    """Test OBV calculation against golden data."""
    # Load test data
    test_data = load_test_data(OBV_TEST_DATA)
    
    # Calculate OBV
    result = calculate_obv(
        close=test_data['close'],
        volume=test_data['volume']
    )
    
    # Compare with expected values
    assert_series_equal(
        result.round(4),
        test_data['obv'].round(4),
        check_names=False,
        check_exact=False,
        rtol=1e-3,
        atol=1e-4
    )

# Test Supertrend
def test_supertrend_calculation():
    """Test Supertrend calculation against golden data."""
    # Load test data
    test_data = load_test_data(SUPERTREND_TEST_DATA)
    
    # Calculate Supertrend
    supertrend, direction = calculate_supertrend(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        period=10,
        multiplier=3.0
    )
    
    # Compare with expected values
    for actual, expected_col in zip(
        [supertrend, direction.astype(float)],
        ['supertrend', 'supertrend_direction']
    ):
        assert_series_equal(
            actual.round(4),
            test_data[expected_col].round(4),
            check_names=False,
            check_exact=False,
            rtol=1e-3,
            atol=1e-4
        )

# Test edge cases
def test_indicators_edge_cases(sample_ohlcv_data):
    """Test indicators with edge cases."""
    df = sample_ohlcv_data
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_rsi(empty_df['close'] if 'close' in empty_df else pd.Series(dtype=float))
    
    # Test with single value
    single_value = df.iloc[:1]
    rsi = calculate_rsi(single_value['close'], period=14)
    assert rsi.isna().all(), "RSI of single value should be NaN"
    
    # Test with all NaN values
    nan_series = pd.Series([np.nan] * 20)
    with pytest.raises(ValueError):
        calculate_rsi(nan_series)
    
    # Test with all zeros
    zero_series = pd.Series([0] * 20)
    rsi = calculate_rsi(zero_series, period=14)
    assert (rsi == 50).all(), "RSI of constant series should be 50"
    
    # Test with constant values
    constant_series = pd.Series([100] * 20)
    rsi = calculate_rsi(constant_series, period=14)
    assert (rsi == 50).all(), "RSI of constant series should be 50"

# Test parameter validation
def test_parameter_validation(sample_ohlcv_data):
    """Test parameter validation for indicator functions."""
    df = sample_ohlcv_data
    
    # Test invalid period (<= 0)
    with pytest.raises(ValueError):
        calculate_rsi(df['close'], period=0)
    
    # Test invalid standard deviation (<= 0)
    with pytest.raises(ValueError):
        calculate_bollinger_bands(df['close'], period=20, num_std=0)
    
    # Test invalid input lengths
    with pytest.raises(ValueError):
        calculate_ichimoku(
            high=df['high'],
            low=df['low'],
            close=df['close'][:-5],  # Mismatched lengths
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52
        )
    
    # Test invalid price data (negative values)
    negative_prices = df['close'].copy()
    negative_prices.iloc[0] = -1
    with pytest.raises(ValueError):
        calculate_rsi(negative_prices)
