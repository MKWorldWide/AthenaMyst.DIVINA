""
Generate golden test data for indicator unit tests.

This script generates test data using a reference implementation (TA-Lib)
and saves it to CSV files for use in unit tests.
"""
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import talib
from tqdm import tqdm

# Ensure the test data directory exists
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Configuration for test data generation
CONFIG = {
    "num_days": 365,  # Number of days of test data to generate
    "start_date": "2022-01-01",
    "seed": 42,  # For reproducible random data
    "base_price": 100.0,
    "volatility": 0.02,  # Daily volatility
    "trend": 0.0005,  # Daily drift/trend
    "volume_mean": 10000,
    "volume_std": 2000,
}


def generate_ohlcv_data() -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic patterns."""
    np.random.seed(CONFIG["seed"])
    
    # Generate date range
    dates = pd.date_range(
        start=CONFIG["start_date"],
        periods=CONFIG["num_days"],
        freq="D",
        tz=timezone.utc,
    )
    
    # Generate random walk for log returns
    log_returns = np.random.normal(
        loc=CONFIG["trend"] - 0.5 * CONFIG["volatility"] ** 2,
        scale=CONFIG["volatility"],
        size=len(dates),
    )
    
    # Convert to prices
    close_prices = CONFIG["base_price"] * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC data with some intraday movement
    # Open: Close from previous day with small gap
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * np.random.uniform(0.99, 1.01)
    
    # High/Low: Add some intraday movement
    intraday_range = close_prices * CONFIG["volatility"] * np.random.uniform(0.5, 2.0, size=len(dates))
    high_prices = np.maximum(open_prices, close_prices) + intraday_range * np.random.uniform(0.5, 1.0, size=len(dates))
    low_prices = np.minimum(open_prices, close_prices) - intraday_range * np.random.uniform(0.5, 1.0, size=len(dates))
    
    # Ensure high > low and high > close, low < close, etc.
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices) * 1.0001)
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices) * 0.9999)
    
    # Generate volume data
    volumes = np.random.normal(
        loc=CONFIG["volume_mean"],
        scale=CONFIG["volume_std"],
        size=len(dates),
    ).astype(int)
    volumes = np.maximum(volumes, 100)  # Ensure minimum volume
    
    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=dates,
    )
    
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ichimoku Cloud indicators to the DataFrame."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = talib.MAX(high, timeperiod=9)
    tenkan_low = talib.MIN(low, timeperiod=9)
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = talib.MAX(high, timeperiod=26)
    kijun_low = talib.MIN(low, timeperiod=26)
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B)
    senkou_high = talib.MAX(high, timeperiod=52)
    senkou_low = talib.MIN(low, timeperiod=52)
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-26)
    
    # Add to DataFrame
    df["tenkan_sen"] = tenkan_sen
    df["kijun_sen"] = kijun_sen
    df["senkou_span_a"] = senkou_span_a
    df["senkou_span_b"] = senkou_span_b
    df["chikou_span"] = chikou_span
    
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI indicator to the DataFrame."""
    df["rsi"] = talib.RSI(df["close"], timeperiod=period)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price (VWAP) to the DataFrame."""
    # VWAP is typically calculated on intraday data, but we'll use daily for testing
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) to the DataFrame."""
    df["atr"] = talib.ATR(
        df["high"],
        df["low"],
        df["close"],
        timeperiod=period,
    )
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands to the DataFrame."""
    upper, middle, lower = talib.BBANDS(
        df["close"],
        timeperiod=period,
        nbdevup=num_std,
        nbdevdn=num_std,
    )
    
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    
    return df


def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Add MACD indicator to the DataFrame."""
    macd, signal, hist = talib.MACD(
        df["close"],
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator to the DataFrame."""
    slowk, slowd = talib.STOCH(
        df["high"],
        df["low"],
        df["close"],
        fastk_period=k_period,
        slowk_period=smooth_k,
        slowk_matype=0,  # Simple moving average
        slowd_period=d_period,
        slowd_matype=0,  # Simple moving average
    )
    
    df["stoch_k"] = slowk
    df["stoch_d"] = slowd
    
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX) to the DataFrame."""
    df["adx"] = talib.ADX(
        df["high"],
        df["low"],
        df["close"],
        timeperiod=period,
    )
    
    # Add +DI and -DI
    df["plus_di"] = talib.PLUS_DI(
        df["high"],
        df["low"],
        df["close"],
        timeperiod=period,
    )
    
    df["minus_di"] = talib.MINUS_DI(
        df["high"],
        df["low"],
        df["close"],
        timeperiod=period,
    )
    
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) to the DataFrame."""
    df["obv"] = talib.OBV(df["close"], df["volume"])
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Add Supertrend indicator to the DataFrame."""
    # Calculate ATR
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    
    # Calculate basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    # Initialize columns
    upper_band = basic_upper.copy()
    lower_band = basic_lower.copy()
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    # Calculate Supertrend
    for i in range(1, len(df)):
        # Update upper band
        if (basic_upper[i] < upper_band[i-1]) or (df["close"][i-1] > upper_band[i-1]):
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i-1]
        
        # Update lower band
        if (basic_lower[i] > lower_band[i-1]) or (df["close"][i-1] < lower_band[i-1]):
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i-1]
        
        # Determine trend direction and Supertrend value
        if df["close"][i] > upper_band[i-1]:
            direction[i] = 1
        elif df["close"][i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if (direction[i] > 0) and (lower_band[i] < lower_band[i-1]):
                lower_band[i] = lower_band[i-1]
            
            if (direction[i] < 0) and (upper_band[i] > upper_band[i-1]):
                upper_band[i] = upper_band[i-1]
        
        # Set Supertrend value
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Add to DataFrame
    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction
    
    return df


def save_test_data(df: pd.DataFrame, filename: str) -> None:
    """Save test data to a CSV file."""
    # Reset index to include date as a column
    df = df.reset_index()
    
    # Round numeric columns to 6 decimal places for consistency
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].round(6)
    
    # Save to CSV
    filepath = TEST_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Saved test data to {filepath}")


def main():
    """Generate all test data files."""
    print("Generating test data...")
    
    # Generate base OHLCV data
    print("Generating OHLCV data...")
    df = generate_ohlcv_data()
    
    # Generate and save test data for each indicator
    print("Generating Ichimoku test data...")
    ichimoku_df = add_ichimoku(df.copy())
    save_test_data(ichimoku_df, "ichimoku_test_data.csv")
    
    print("Generating RSI test data...")
    rsi_df = add_rsi(df.copy())
    save_test_data(rsi_df[["close", "rsi"]], "rsi_test_data.csv")
    
    print("Generating VWAP test data...")
    vwap_df = add_vwap(df.copy())
    save_test_data(vwap_df[["high", "low", "close", "volume", "vwap"]], "vwap_test_data.csv")
    
    print("Generating ATR test data...")
    atr_df = add_atr(df.copy())
    save_test_data(atr_df[["high", "low", "close", "atr"]], "atr_test_data.csv")
    
    print("Generating Bollinger Bands test data...")
    bb_df = add_bollinger_bands(df.copy())
    save_test_data(bb_df[["close", "bb_upper", "bb_middle", "bb_lower"]], "bbands_test_data.csv")
    
    print("Generating MACD test data...")
    macd_df = add_macd(df.copy())
    save_test_data(
        macd_df[["close", "macd", "macd_signal", "macd_hist"]],
        "macd_test_data.csv"
    )
    
    print("Generating Stochastic test data...")
    stoch_df = add_stochastic(df.copy())
    save_test_data(
        stoch_df[["high", "low", "close", "stoch_k", "stoch_d"]],
        "stochastic_test_data.csv"
    )
    
    print("Generating ADX test data...")
    adx_df = add_adx(df.copy())
    save_test_data(
        adx_df[["high", "low", "close", "adx", "plus_di", "minus_di"]],
        "adx_test_data.csv"
    )
    
    print("Generating OBV test data...")
    obv_df = add_obv(df.copy())
    save_test_data(
        obv_df[["close", "volume", "obv"]],
        "obv_test_data.csv"
    )
    
    print("Generating Supertrend test data...")
    supertrend_df = add_supertrend(df.copy())
    save_test_data(
        supertrend_df[["high", "low", "close", "supertrend", "supertrend_direction"]],
        "supertrend_test_data.csv"
    )
    
    print("\nTest data generation complete!")


if __name__ == "__main__":
    main()
