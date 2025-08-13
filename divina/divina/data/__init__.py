""
Data management module for AthenaMyst:Divina.

Handles data fetching, caching, and multi-timeframe aggregation.
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from ..models import Candle, Timeframe, Signal
from ..connectors import create_connector, BaseConnector
from ..config import settings
from ..indicators import Indicators


@dataclass
class TimeframeData:
    """Container for market data at a specific timeframe."""
    pair: str
    timeframe: Timeframe
    candles: List[Candle]
    indicators: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize indicators if not provided."""
        if self.indicators is None:
            self.indicators = {}
    
    @property
    def last_candle(self) -> Optional[Candle]:
        """Get the most recent candle."""
        return self.candles[-1] if self.candles else None
    
    @property
    def is_stale(self) -> bool:
        """Check if the data is stale based on the timeframe."""
        if not self.candles:
            return True
            
        last_candle = self.candles[-1]
        now = datetime.utcnow()
        
        # Define max age based on timeframe
        timeframe_minutes = {
            Timeframe.M1: 2,      # 2 minutes
            Timeframe.M5: 10,     # 10 minutes
            Timeframe.M15: 30,    # 30 minutes
            Timeframe.M30: 60,    # 1 hour
            Timeframe.H1: 120,    # 2 hours
            Timeframe.H4: 300,    # 5 hours
            Timeframe.D1: 3600,   # 1 day
            Timeframe.W1: 86400,  # 1 week
            Timeframe.MN: 259200  # 3 days
        }.get(self.timeframe, 300)  # Default to 5 minutes
        
        return (now - last_candle.timestamp).total_seconds() > (timeframe_minutes * 60)


class DataManager:
    """Manages market data fetching, caching, and multi-timeframe aggregation."""
    
    def __init__(self, connector: Optional[BaseConnector] = None):
        """Initialize the data manager."""
        self.connector = connector or create_connector(
            'oanda',
            api_key=settings.oanda.api_key,
            account_id=settings.oanda.account_id,
            environment=settings.oanda.environment
        )
        self.cache_dir = Path(settings.data_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeframe_data: Dict[Tuple[str, Timeframe], TimeframeData] = {}
        self.historical_data: Dict[Tuple[str, Timeframe], List[Candle]] = {}
    
    async def initialize(self) -> None:
        """Initialize the data manager and load initial data."""
        # Load initial data for all configured pairs and timeframes
        tasks = []
        
        for pair in settings.trading.pairs:
            # Load signal timeframe
            tasks.append(
                self.load_timeframe_data(
                    pair,
                    Timeframe(settings.trading.signal_tf),
                    lookback_bars=500  # Load enough data for indicators
                )
            )
            
            # Load confirmation timeframe
            if settings.trading.confirm_tf != settings.trading.signal_tf:
                tasks.append(
                    self.load_timeframe_data(
                        pair,
                        Timeframe(settings.trading.confirm_tf),
                        lookback_bars=200
                    )
                )
        
        # Run all data loading in parallel
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def load_timeframe_data(
        self,
        pair: str,
        timeframe: Timeframe,
        lookback_bars: int = 100,
        force_refresh: bool = False
    ) -> TimeframeData:
        """Load or refresh data for a specific pair and timeframe."""
        cache_key = (pair, timeframe)
        
        # Check if we already have fresh data
        if not force_refresh and cache_key in self.timeframe_data:
            data = self.timeframe_data[cache_key]
            if not data.is_stale:
                return data
        
        # Calculate time range for historical data
        end_time = datetime.utcnow()
        start_time = self._calculate_start_time(timeframe, lookback_bars, end_time)
        
        # Fetch historical data
        candles = await self.connector.fetch_historical_data(
            pair=pair,
            timeframe=timeframe,
            start=start_time,
            end=end_time,
            use_cache=not force_refresh
        )
        
        # Calculate indicators
        indicators = Indicators.calculate_all(candles)
        
        # Create and store TimeframeData
        data = TimeframeData(
            pair=pair,
            timeframe=timeframe,
            candles=candles,
            indicators=indicators
        )
        
        self.timeframe_data[cache_key] = data
        return data
    
    def _calculate_start_time(
        self,
        timeframe: Timeframe,
        lookback_bars: int,
        end_time: datetime
    ) -> datetime:
        """Calculate the start time for historical data based on lookback bars."""
        # Approximate minutes per bar
        minutes_per_bar = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
            Timeframe.W1: 10080,
            Timeframe.MN: 43200,  # Approximate
        }.get(timeframe, 60)
        
        # Add buffer for weekends/holidays
        buffer_factor = 1.5
        total_minutes = lookback_bars * minutes_per_bar * buffer_factor
        
        return end_time - timedelta(minutes=total_minutes)
    
    async def update_all(self) -> None:
        """Update all loaded timeframe data."""
        tasks = []
        
        for (pair, timeframe), data in self.timeframe_data.items():
            if data.is_stale:
                tasks.append(
                    self.load_timeframe_data(
                        pair,
                        timeframe,
                        lookback_bars=10,  # Just get the latest few candles
                        force_refresh=True
                    )
                )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_timeframe_data(
        self,
        pair: str,
        timeframe: Timeframe
    ) -> Optional[TimeframeData]:
        """Get data for a specific pair and timeframe."""
        return self.timeframe_data.get((pair, timeframe))
    
    def get_multi_timeframe_data(
        self,
        pair: str,
        timeframes: List[Timeframe]
    ) -> Dict[Timeframe, TimeframeData]:
        """Get data for multiple timeframes for a pair."""
        return {
            tf: self.timeframe_data.get((pair, tf))
            for tf in timeframes
            if (pair, tf) in self.timeframe_data
        }
    
    async def get_historical_data(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Candle]:
        """Get historical data with caching."""
        cache_key = (pair, timeframe, start, end or datetime.utcnow())
        
        if use_cache and cache_key in self.historical_data:
            return self.historical_data[cache_key]
        
        # Fetch from the connector
        candles = await self.connector.fetch_historical_data(
            pair=pair,
            timeframe=timeframe,
            start=start,
            end=end,
            use_cache=use_cache
        )
        
        if use_cache:
            self.historical_data[cache_key] = candles
        
        return candles
    
    async def stream_updates(self) -> None:
        """Stream real-time updates for all active pairs and timeframes."""
        # TODO: Implement WebSocket streaming
        # For now, we'll just poll for updates
        while True:
            try:
                await self.update_all()
                await asyncio.sleep(60)  # Poll every minute
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def get_signal_data(
        self,
        signal: Signal
    ) -> Dict[str, Any]:
        """Get relevant data for a trading signal."""
        data = {
            'pair': signal.pair,
            'timeframe': signal.timeframe,
            'direction': signal.direction,
            'strength': signal.strength,
            'price': signal.price,
            'timestamp': signal.timestamp,
            'indicators': signal.indicators,
            'metadata': signal.metadata or {}
        }
        
        # Add additional context from timeframe data
        tf_data = self.get_timeframe_data(signal.pair, signal.timeframe)
        if tf_data and tf_data.candles:
            last_candle = tf_data.candles[-1]
            data.update({
                'open': last_candle.open,
                'high': last_candle.high,
                'low': last_candle.low,
                'close': last_candle.close,
                'volume': last_candle.volume,
            })
        
        return data
