""
Enhanced DataManager with async I/O, retries, and caching.
"""
import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from loguru import logger

from ..connectors import BaseConnector, create_connector
from ..models import Candle, Timeframe, TimeframeData, TimeframeLike
from ..utils.cache import HybridCache
from ..utils.retry import async_retry
from ..config import settings
from ..indicators import Indicators


class DataManager:
    """Manages market data with caching and async I/O."""
    
    def __init__(
        self,
        connector: Optional[BaseConnector] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True,
        memory_cache_size: int = 1000,
        memory_cache_ttl: float = 300.0,  # 5 minutes
        disk_cache_ttl: float = 86400.0,  # 24 hours
        disk_cache_size_mb: float = 100.0,  # 100 MB
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff: float = 2.0,
    ):
        """Initialize the DataManager.
        
        Args:
            connector: Data connector to use (defaults to OANDA)
            cache_dir: Directory for disk cache (defaults to settings.data_dir/cache)
            enable_disk_cache: Whether to enable disk caching
            enable_memory_cache: Whether to enable in-memory caching
            memory_cache_size: Maximum number of items in memory cache
            memory_cache_ttl: TTL for memory cache entries in seconds
            disk_cache_ttl: TTL for disk cache entries in seconds
            disk_cache_size_mb: Maximum size of disk cache in MB
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            backoff: Backoff multiplier for retries
        """
        self.connector = connector or create_connector(
            'oanda',
            api_key=settings.oanda.api_key,
            account_id=settings.oanda.account_id,
            environment=settings.oanda.environment
        )
        
        # Set up cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.data_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache = HybridCache(
            cache_dir=self.cache_dir / 'market_data',
            memory_max_size=memory_cache_size if enable_memory_cache else 0,
            memory_ttl=memory_cache_ttl,
            disk_ttl=disk_cache_ttl if enable_disk_cache else 0,
            disk_max_size_mb=disk_cache_size_mb,
            compress=True
        )
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff = backoff
        
        # In-memory data store
        self.timeframe_data: Dict[Tuple[str, Timeframe], TimeframeData] = {}
        self.last_updated: Dict[Tuple[str, Timeframe], datetime] = {}
        
        # Locks for thread safety
        self._locks: Dict[Tuple[str, Timeframe], asyncio.Lock] = defaultdict(asyncio.Lock)
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the data manager and load initial data."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:  # Double-check
                return
            
            logger.info("Initializing DataManager...")
            
            # Load initial data for all configured pairs and timeframes
            tasks = []
            timeframes = set()
            
            # Add signal and confirmation timeframes
            timeframes.add(Timeframe(settings.trading.signal_tf))
            timeframes.add(Timeframe(settings.trading.confirm_tf))
            
            # Create tasks for loading initial data
            for pair in settings.trading.pairs:
                for timeframe in timeframes:
                    tasks.append(
                        self.load_timeframe_data(
                            pair=pair,
                            timeframe=timeframe,
                            lookback_bars=500  # Load enough data for indicators
                        )
                    )
            
            # Run all data loading in parallel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Start background tasks
            asyncio.create_task(self._periodic_cleanup())
            
            self._initialized = True
            logger.info("DataManager initialized")
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old data and cache."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def cleanup(self) -> None:
        """Clean up old data and cache."""
        logger.debug("Cleaning up old data and cache...")
        
        # Clean up cache
        await self.cache.cleanup()
        
        # Clean up old timeframes that haven't been accessed in a while
        now = datetime.now(timezone.utc)
        stale_keys = []
        
        for (pair, tf), last_updated in list(self.last_updated.items()):
            if (now - last_updated).total_seconds() > 86400:  # 24 hours
                stale_keys.append((pair, tf))
        
        for key in stale_keys:
            if key in self.timeframe_data:
                del self.timeframe_data[key]
            if key in self.last_updated:
                del self.last_updated[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale timeframes")
    
    def _get_cache_key(
        self,
        pair: str,
        timeframe: Timeframe,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> str:
        """Generate a cache key for the given parameters."""
        key_parts = [
            'candles',
            pair,
            str(timeframe),
            start.isoformat() if start else '',
            end.isoformat() if end else '',
            str(limit) if limit is not None else ''
        ]
        return ':'.join(key_parts)
    
    async def load_timeframe_data(
        self,
        pair: str,
        timeframe: TimeframeLike,
        lookback_bars: int = 100,
        force_refresh: bool = False,
        use_cache: bool = True
    ) -> Optional[TimeframeData]:
        """Load or refresh data for a specific pair and timeframe."""
        if isinstance(timeframe, str):
            timeframe = Timeframe(timeframe)
        
        cache_key = self._get_cache_key(
            pair=pair,
            timeframe=timeframe,
            limit=lookback_bars
        )
        
        # Check if we already have fresh data in memory
        now = datetime.now(timezone.utc)
        data_key = (pair, timeframe)
        
        if (not force_refresh and 
            data_key in self.timeframe_data and 
            (now - self.last_updated.get(data_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < 60):
            return self.timeframe_data[data_key]
        
        # Use a lock for this specific pair+timeframe to prevent concurrent updates
        async with self._locks[data_key]:
            # Check again after acquiring the lock
            if (not force_refresh and 
                data_key in self.timeframe_data and 
                (now - self.last_updated.get(data_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < 60):
                return self.timeframe_data[data_key]
            
            # Try to get from cache first
            cached_data = None
            if use_cache and not force_refresh:
                try:
                    cached_data = await self.cache.get(cache_key)
                    if cached_data:
                        # Convert dict back to TimeframeData
                        candles = [Candle(**c) for c in cached_data.get('candles', [])]
                        indicators = cached_data.get('indicators', {})
                        
                        if candles:
                            tf_data = TimeframeData(
                                pair=pair,
                                timeframe=timeframe,
                                candles=candles,
                                indicators=indicators
                            )
                            
                            self.timeframe_data[data_key] = tf_data
                            self.last_updated[data_key] = now
                            
                            logger.debug(f"Loaded {len(candles)} {pair} {timeframe} candles from cache")
                            return tf_data
                
                except Exception as e:
                    logger.warning(f"Error loading from cache: {e}")
            
            # If we get here, we need to fetch fresh data
            logger.debug(f"Fetching fresh data for {pair} {timeframe}...")
            
            try:
                # Calculate time range
                end_time = datetime.now(timezone.utc)
                start_time = self._calculate_start_time(timeframe, lookback_bars, end_time)
                
                # Fetch candles with retry
                candles = await self._fetch_candles_with_retry(
                    pair=pair,
                    timeframe=timeframe,
                    start=start_time,
                    end=end_time,
                    limit=lookback_bars
                )
                
                if not candles:
                    logger.warning(f"No candles returned for {pair} {timeframe}")
                    return None
                
                # Calculate indicators
                indicators = Indicators.calculate_all(candles)
                
                # Create TimeframeData
                tf_data = TimeframeData(
                    pair=pair,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=indicators
                )
                
                # Update cache
                if use_cache:
                    try:
                        cache_data = {
                            'candles': [c.dict() for c in candles],
                            'indicators': indicators,
                            'updated_at': now.isoformat()
                        }
                        await self.cache.set(cache_key, cache_data)
                    except Exception as e:
                        logger.warning(f"Error updating cache: {e}")
                
                # Update in-memory store
                self.timeframe_data[data_key] = tf_data
                self.last_updated[data_key] = now
                
                logger.info(f"Loaded {len(candles)} {pair} {timeframe} candles")
                return tf_data
            
            except Exception as e:
                logger.error(f"Error loading {pair} {timeframe}: {e}")
                return None
    
    async def _fetch_candles_with_retry(
        self,
        pair: str,
        timeframe: Timeframe,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """Fetch candles with retry logic."""
        async def _fetch() -> List[Candle]:
            return await self.connector.fetch_historical_data(
                pair=pair,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                use_cache=False  # We handle caching ourselves
            )
        
        try:
            return await async_retry(
                _fetch,
                max_retries=self.max_retries,
                initial_delay=self.retry_delay,
                max_delay=self.max_retry_delay,
                backoff=self.backoff,
                log_retries=True
            )
        except Exception as e:
            logger.error(f"Failed to fetch candles for {pair} {timeframe} after retries: {e}")
            return []
    
    def _calculate_start_time(
        self,
        timeframe: Timeframe,
        lookback_bars: int,
        end_time: datetime
    ) -> datetime:
        """Calculate the start time based on lookback bars and timeframe."""
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
        
        # Add buffer for weekends/holidays (30% more time)
        buffer_factor = 1.3
        total_minutes = lookback_bars * minutes_per_bar * buffer_factor
        
        return end_time - timedelta(minutes=total_minutes)
    
    async def get_timeframe_data(
        self,
        pair: str,
        timeframe: TimeframeLike,
        lookback_bars: int = 100,
        force_refresh: bool = False
    ) -> Optional[TimeframeData]:
        """Get data for a specific pair and timeframe."""
        if isinstance(timeframe, str):
            timeframe = Timeframe(timeframe)
        
        data_key = (pair, timeframe)
        
        # Check if we have fresh data in memory
        now = datetime.now(timezone.utc)
        
        if (not force_refresh and 
            data_key in self.timeframe_data and 
            (now - self.last_updated.get(data_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < 60):
            return self.timeframe_data[data_key]
        
        # Otherwise, load the data
        return await self.load_timeframe_data(
            pair=pair,
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            force_refresh=force_refresh
        )
    
    async def get_multi_timeframe_data(
        self,
        pair: str,
        timeframes: List[TimeframeLike],
        lookback_bars: int = 100,
        force_refresh: bool = False
    ) -> Dict[Timeframe, TimeframeData]:
        """Get data for multiple timeframes for a pair."""
        # Convert string timeframes to Timeframe objects
        tf_objs = [Timeframe(tf) if isinstance(tf, str) else tf for tf in timeframes]
        
        # Create tasks for loading each timeframe
        tasks = [
            self.get_timeframe_data(
                pair=pair,
                timeframe=tf,
                lookback_bars=lookback_bars,
                force_refresh=force_refresh
            )
            for tf in tf_objs
        ]
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data = {}
        for tf, result in zip(tf_objs, results):
            if isinstance(result, Exception):
                logger.error(f"Error loading {pair} {tf}: {result}")
            elif result is not None:
                data[tf] = result
        
        return data
    
    async def update_all(self) -> None:
        """Update all loaded timeframe data."""
        if not self.timeframe_data:
            return
        
        logger.debug(f"Updating {len(self.timeframe_data)} timeframes...")
        
        # Create tasks for updating each timeframe
        tasks = []
        for (pair, tf), data in self.timeframe_data.items():
            tasks.append(
                self.load_timeframe_data(
                    pair=pair,
                    timeframe=tf,
                    lookback_bars=len(data.candles) if data.candles else 100,
                    force_refresh=True
                )
            )
        
        # Run updates in parallel
        await asyncio.gather(*tasks, return_exceptions=True)
    
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
    
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'connector') and self.connector:
            await self.connector.close()
        
        # Clear caches
        await self.cache.clear()
        
        # Clear in-memory data
        self.timeframe_data.clear()
        self.last_updated.clear()
        
        logger.info("DataManager closed")
