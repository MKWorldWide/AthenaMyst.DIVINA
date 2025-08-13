""
Unit tests for the DataManager class.

This module contains tests for data management functionality including
market data updates, retrieval, and indicator calculations.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from divina.data.manager import DataManager
from divina.indicators import calculate_ichimoku, calculate_rsi

# Test configuration
TEST_SYMBOL = "EURUSD"
TEST_TIMEFRAMES = ["1h", "4h", "1d"]
TEST_START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
TEST_END_DATE = datetime(2023, 1, 31, tzinfo=timezone.utc)


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, pd.DataFrame]:
    """Create sample OHLCV data for multiple timeframes."""
    np.random.seed(42)
    data = {}
    
    # Generate base 1h data
    dates_1h = pd.date_range(
        start=TEST_START_DATE,
        end=TEST_END_DATE,
        freq="1h",
        tz=timezone.utc
    )
    
    # Generate random walk for close prices
    n_bars = len(dates_1h)
    close_1h = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0005)
    
    # Create OHLCV data with some structure
    data["1h"] = pd.DataFrame({
        'open': close_1h - np.abs(np.random.randn(n_bars) * 0.0005),
        'high': close_1h + np.abs(np.random.randn(n_bars) * 0.0005),
        'low': close_1h - np.abs(np.random.randn(n_bars) * 0.0005),
        'close': close_1h,
        'volume': (np.random.rand(n_bars) * 1000).astype(int) + 100,
    }, index=dates_1h)
    
    # Generate 4h data by resampling 1h data
    data["4h"] = data["1h"].resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Generate 1d data by resampling 1h data
    data["1d"] = data["1h"].resample('1d').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return data


def test_data_manager_initialization():
    """Test DataManager initialization."""
    # Initialize with default parameters
    dm = DataManager()
    
    # Check default attributes
    assert dm.symbols == set()
    assert dm.timeframes == set()
    assert dm.data == {}
    assert dm.indicators == {}
    
    # Initialize with symbols and timeframes
    dm = DataManager(symbols=["EURUSD", "GBPUSD"], timeframes=["1h", "4h"])
    assert dm.symbols == {"EURUSD", "GBPUSD"}
    assert dm.timeframes == {"1h", "4h"}


def test_update_market_data(sample_ohlcv_data):
    """Test updating market data."""
    dm = DataManager()
    
    # Update with 1h data
    dm.update_market_data(TEST_SYMBOL, "1h", sample_ohlcv_data["1h"])
    
    # Check data was added
    assert TEST_SYMBOL in dm.symbols
    assert "1h" in dm.timeframes
    assert TEST_SYMBOL in dm.data
    assert "1h" in dm.data[TEST_SYMBOL]
    
    # Check data integrity
    assert_frame_equal(
        dm.data[TEST_SYMBOL]["1h"][['open', 'high', 'low', 'close', 'volume']],
        sample_ohlcv_data["1h"]
    )
    
    # Update with 4h data
    dm.update_market_data(TEST_SYMBOL, "4h", sample_ohlcv_data["4h"])
    assert "4h" in dm.timeframes
    assert "4h" in dm.data[TEST_SYMBOL]
    
    # Update with partial data (should merge with existing)
    partial_data = sample_ohlcv_data["1h"].iloc[-10:].copy()
    dm.update_market_data(TEST_SYMBOL, "1h", partial_data)
    assert len(dm.data[TEST_SYMBOL]["1h"]) == len(sample_ohlcv_data["1h"])


def test_get_market_data(sample_ohlcv_data):
    """Test retrieving market data."""
    dm = DataManager()
    
    # Add test data
    for tf, df in sample_ohlcv_data.items():
        dm.update_market_data(TEST_SYMBOL, tf, df)
    
    # Get full data
    for tf in sample_ohlcv_data.keys():
        df = dm.get_market_data(TEST_SYMBOL, tf)
        assert_frame_equal(df, sample_ohlcv_data[tf])
    
    # Get data with limit
    limit = 10
    for tf in sample_ohlcv_data.keys():
        df = dm.get_market_data(TEST_SYMBOL, tf, limit=limit)
        assert len(df) == min(limit, len(sample_ohlcv_data[tf]))
        assert df.index[0] == sample_ohlcv_data[tf].index[-limit]
    
    # Get data with date range
    start_date = TEST_START_DATE + timedelta(days=5)
    end_date = TEST_START_DATE + timedelta(days=10)
    
    for tf in sample_ohlcv_data.keys():
        df = dm.get_market_data(
            TEST_SYMBOL, 
            tf, 
            start_date=start_date,
            end_date=end_date
        )
        assert df.index[0] >= start_date
        assert df.index[-1] <= end_date


def test_add_technical_indicator(sample_ohlcv_data):
    """Test adding technical indicators."""
    dm = DataManager()
    dm.update_market_data(TEST_SYMBOL, "1h", sample_ohlcv_data["1h"])
    
    # Add RSI indicator
    dm.add_technical_indicator(
        symbol=TEST_SYMBOL,
        timeframe="1h",
        indicator_name="rsi",
        indicator_func=calculate_rsi,
        period=14
    )
    
    # Check indicator was added
    assert "rsi" in dm.indicators.get(TEST_SYMBOL, {}).get("1h", {})
    
    # Get data with indicator
    df = dm.get_market_data(TEST_SYMBOL, "1h", include_indicators=True)
    assert "rsi" in df.columns
    assert not df["rsi"].isnull().all()
    
    # Add Ichimoku indicator with multiple outputs
    dm.add_technical_indicator(
        symbol=TEST_SYMBOL,
        timeframe="1h",
        indicator_name="ichimoku",
        indicator_func=calculate_ichimoku,
        high=dm.data[TEST_SYMBOL]["1h"]["high"],
        low=dm.data[TEST_SYMBOL]["1h"]["low"],
        close=dm.data[TEST_SYMBOL]["1h"]["close"],
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52
    )
    
    # Check Ichimoku components were added
    ichimoku_cols = [
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 
        'senkou_span_b', 'chikou_span'
    ]
    
    df = dm.get_market_data(TEST_SYMBOL, "1h", include_indicators=True)
    for col in ichimoku_cols:
        assert col in df.columns


def test_get_latest_data(sample_ohlcv_data):
    """Test getting the latest market data."""
    dm = DataManager()
    
    # Add test data
    for tf, df in sample_ohlcv_data.items():
        dm.update_market_data(TEST_SYMBOL, tf, df)
    
    # Get latest data for all timeframes
    latest = dm.get_latest_data(TEST_SYMBOL)
    
    # Check structure
    assert isinstance(latest, dict)
    assert set(latest.keys()) == set(sample_ohlcv_data.keys())
    
    # Check values
    for tf, df in latest.items():
        assert isinstance(df, pd.Series)
        assert df.name == sample_ohlcv_data[tf].index[-1]
        assert df["close"] == sample_ohlcv_data[tf]["close"].iloc[-1]


def test_resample_data(sample_ohlcv_data):
    """Test resampling market data."""
    dm = DataManager()
    
    # Add 1h data
    dm.update_market_data(TEST_SYMBOL, "1h", sample_ohlcv_data["1h"])
    
    # Resample to 4h
    resampled = dm.resample_data(
        symbol=TEST_SYMBOL,
        source_timeframe="1h",
        target_timeframe="4h"
    )
    
    # Check resampled data
    assert isinstance(resampled, pd.DataFrame)
    assert not resampled.empty
    
    # Should be approximately 1/4 the number of 1h bars (minus some for partial periods)
    expected_length = len(sample_ohlcv_data["1h"].resample('4h').count().dropna())
    assert abs(len(resampled) - expected_length) <= 1
    
    # Check OHLCV aggregation
    assert (resampled["high"] >= resampled["low"]).all()
    assert (resampled["high"] >= resampled["close"]).all()
    assert (resampled["high"] >= resampled["open"]).all()
    assert (resampled["low"] <= resampled["close"]).all()
    assert (resampled["low"] <= resampled["open"]).all()
    assert (resampled["volume"] >= 0).all()


def test_clean_old_data(sample_ohlcv_data):
    """Test cleaning old market data."""
    dm = DataManager()
    
    # Add test data
    for tf, df in sample_ohlcv_data.items():
        dm.update_market_data(TEST_SYMBOL, tf, df)
    
    # Get original lengths
    original_lengths = {
        tf: len(df) for tf, df in dm.data[TEST_SYMBOL].items()
    }
    
    # Clean data older than 15 days from the end
    cutoff_date = TEST_END_DATE - timedelta(days=15)
    dm.clean_old_data(cutoff_date=cutoff_date)
    
    # Check data was cleaned
    for tf in sample_ohlcv_data.keys():
        df = dm.get_market_data(TEST_SYMBOL, tf)
        assert df.index[0] >= cutoff_date
        assert len(df) < original_lengths[tf]


def test_save_and_load_data(tmp_path, sample_ohlcv_data):
    """Test saving and loading market data to/from disk."""
    # Create a temporary directory
    data_dir = tmp_path / "market_data"
    data_dir.mkdir()
    
    # Initialize and populate DataManager
    dm1 = DataManager()
    for tf, df in sample_ohlcv_data.items():
        dm1.update_market_data(TEST_SYMBOL, tf, df)
    
    # Save data
    dm1.save_data(data_dir)
    
    # Check files were created
    for tf in sample_ohlcv_data.keys():
        file_path = data_dir / f"{TEST_SYMBOL}_{tf}.parquet"
        assert file_path.exists()
    
    # Load into a new DataManager
    dm2 = DataManager()
    dm2.load_data(data_dir)
    
    # Check data was loaded correctly
    assert dm1.symbols == dm2.symbols
    assert dm1.timeframes == dm2.timeframes
    
    for symbol in dm1.symbols:
        for tf in dm1.timeframes:
            df1 = dm1.get_market_data(symbol, tf)
            df2 = dm2.get_market_data(symbol, tf)
            assert_frame_equal(df1, df2)
