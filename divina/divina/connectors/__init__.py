""
Data connectors for AthenaMyst:Divina.

This module provides interfaces to various data sources like OANDA and Kraken.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from ..models import Candle, Timeframe
from ..config import settings


class BaseConnector(ABC):
    """Base class for all data connectors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the connector with configuration."""
        self.config = config or {}
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()
        self.rate_limit_remaining = 100  # Default rate limit
        self.rate_limit_reset = 0  # Timestamp when rate limit resets
    
    def _create_session(self) -> httpx.AsyncClient:
        """Create an HTTP session with appropriate headers and timeouts."""
        return httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'AthenaMyst-Divina/0.1.0',
                'Accept': 'application/json',
            },
            follow_redirects=True,
        )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if hasattr(self, 'session') and self.session:
            await self.session.aclose()
    
    def _get_cache_key(self, pair: str, timeframe: Timeframe, start: datetime, end: datetime) -> str:
        """Generate a cache key for the given parameters."""
        return f"{self.__class__.__name__.lower()}_{pair}_{timeframe}_{start.isoformat()}_{end.isoformat()}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Candle]]:
        """Load data from cache if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            # Check if cache is expired (1 hour by default)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > self.config.get('cache_ttl', 3600):
                return None
                
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return [Candle(**candle) for candle in data]
                
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load from cache {cache_file}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, candles: List[Candle]) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump([candle.dict() for candle in candles], f, default=str)
        except OSError as e:
            logger.warning(f"Failed to save to cache {cache_file}: {e}")
    
    async def _handle_rate_limit(self) -> None:
        """Handle rate limiting by sleeping if necessary."""
        now = time.time()
        
        if self.rate_limit_remaining <= 0 and now < self.rate_limit_reset:
            sleep_time = self.rate_limit_reset - now + 1  # Add 1 second buffer
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
    
    @abstractmethod
    async def fetch_historical_data(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Candle]:
        """Fetch historical candle data."""
        pass
    
    @abstractmethod
    async def fetch_latest_data(
        self,
        pair: str,
        timeframe: Timeframe,
        limit: int = 100
    ) -> List[Candle]:
        """Fetch the most recent candle data."""
        pass
    
    @abstractmethod
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """Get list of available trading instruments."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass


class OandaConnector(BaseConnector):
    """OANDA API connector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OANDA connector."""
        super().__init__(config)
        
        # Required configuration
        self.api_key = config.get('api_key') or settings.oanda.api_key
        self.account_id = config.get('account_id') or settings.oanda.account_id
        self.environment = config.get('environment', 'practice')
        
        # API endpoints
        self.base_url = f"https://api-fx{'trade' if self.environment == 'live' else 'practice'}.oanda.com/v3"
        self.stream_url = f"https://stream-fx{'trade' if self.environment == 'live' else 'practice'}.oanda.com/v3"
        
        # Configure session with auth headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        })
        
        # Timeframe mapping (OANDA format)
        self.timeframe_map = {
            Timeframe.M1: 'M1',
            Timeframe.M5: 'M5',
            Timeframe.M15: 'M15',
            Timeframe.M30: 'M30',
            Timeframe.H1: 'H1',
            Timeframe.H4: 'H4',
            Timeframe.D1: 'D',
            Timeframe.W1: 'W',
            Timeframe.MN: 'M',
        }
    
    def _map_timeframe(self, timeframe: Timeframe) -> str:
        """Map standard timeframe to OANDA format."""
        return self.timeframe_map.get(timeframe, str(timeframe))
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """Make an HTTP request with retries and rate limiting."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(retries):
            try:
                # Check rate limits
                await self._handle_rate_limit()
                
                # Make the request
                response = await self.session.request(
                    method,
                    url,
                    params=params,
                    json=data
                )
                
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('RateLimit-Remaining', 100))
                self.rate_limit_reset = int(response.headers.get('RateLimit-Reset', 0))
                
                # Handle response
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    retry_after = int(e.response.headers.get('Retry-After', '5'))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                    
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                if attempt == retries - 1:
                    raise
                    
            except (httpx.RequestError, json.JSONDecodeError) as e:
                logger.error(f"Request failed: {e}")
                if attempt == retries - 1:
                    raise
                
            # Exponential backoff
            time.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {retries} attempts")
    
    async def fetch_historical_data(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Candle]:
        """Fetch historical candle data from OANDA."""
        end = end or datetime.utcnow()
        cache_key = self._get_cache_key(pair, timeframe, start, end)
        
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded {len(cached_data)} candles from cache for {pair} {timeframe}")
                return cached_data
        
        # Fetch from API
        logger.info(f"Fetching historical data for {pair} {timeframe} from {start} to {end}")
        
        # OANDA uses RFC3339 timestamps
        start_time = start.isoformat('T') + 'Z'
        end_time = end.isoformat('T') + 'Z'
        
        params = {
            'from': start_time,
            'to': end_time,
            'granularity': self._map_timeframe(timeframe),
            'price': 'BA',  # Bid/Ask
            'count': 5000,  # Max candles per request
        }
        
        all_candles = []
        
        try:
            # OANDA returns data in pages, so we need to handle pagination
            while True:
                response = await self._make_request(
                    'GET',
                    f"/instruments/{pair}/candles",
                    params=params
                )
                
                candles = response.get('candles', [])
                if not candles:
                    break
                
                # Convert to our Candle model
                for candle_data in candles:
                    if not candle_data.get('complete', False):
                        continue  # Skip incomplete candles
                        
                    mid = candle_data.get('mid', {})
                    if not mid:
                        continue
                        
                    candle = Candle(
                        timestamp=pd.to_datetime(candle_data['time']),
                        open=float(mid.get('o', 0)),
                        high=float(mid.get('h', 0)),
                        low=float(mid.get('l', 0)),
                        close=float(mid.get('c', 0)),
                        volume=int(candle_data.get('volume', 0))
                    )
                    all_candles.append(candle)
                
                # Check if we have more data
                if len(candles) < 5000:
                    break
                    
                # Update params for next page
                last_candle = candles[-1]
                params['from'] = last_candle['time']
            
            # Save to cache
            if all_candles and use_cache:
                self._save_to_cache(cache_key, all_candles)
            
            return all_candles
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    async def fetch_latest_data(
        self,
        pair: str,
        timeframe: Timeframe,
        limit: int = 100
    ) -> List[Candle]:
        """Fetch the most recent candle data."""
        try:
            response = await self._make_request(
                'GET',
                f"/instruments/{pair}/candles",
                params={
                    'granularity': self._map_timeframe(timeframe),
                    'price': 'BA',
                    'count': limit,
                }
            )
            
            candles = []
            for candle_data in response.get('candles', []):
                if not candle_data.get('complete', False):
                    continue
                    
                mid = candle_data.get('mid', {})
                if not mid:
                    continue
                    
                candle = Candle(
                    timestamp=pd.to_datetime(candle_data['time']),
                    open=float(mid.get('o', 0)),
                    high=float(mid.get('h', 0)),
                    low=float(mid.get('l', 0)),
                    close=float(mid.get('c', 0)),
                    volume=int(candle_data.get('volume', 0))
                )
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch latest data: {e}")
            raise
    
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """Get list of available trading instruments."""
        try:
            response = await self._make_request(
                'GET',
                f"/accounts/{self.account_id}/instruments"
            )
            return response.get('instruments', [])
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            response = await self._make_request(
                'GET',
                f"/accounts/{self.account_id}"
            )
            return response.get('account', {})
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise


class KrakenConnector(BaseConnector):
    """Kraken API connector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Kraken connector."""
        super().__init__(config)
        
        # Required configuration
        self.api_key = config.get('api_key') or settings.kraken.api_key
        self.api_secret = config.get('api_secret') or settings.kraken.api_secret
        
        # API endpoints
        self.base_url = "https://api.kraken.com/0"
        
        # Timeframe mapping (Kraken format in minutes)
        self.timeframe_map = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
            Timeframe.W1: 10080,
            Timeframe.MN: 21600,  # Kraken's closest to monthly
        }
    
    def _map_timeframe(self, timeframe: Timeframe) -> int:
        """Map standard timeframe to Kraken format."""
        return self.timeframe_map.get(timeframe, 60)  # Default to 1h
    
    async def _make_public_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """Make a public API request."""
        url = f"{self.base_url}/public/{endpoint}"
        
        for attempt in range(retries):
            try:
                response = await self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    raise Exception(f"Kraken API error: {data['error']}")
                    
                return data
                
            except (httpx.RequestError, json.JSONDecodeError) as e:
                logger.error(f"Request failed: {e}")
                if attempt == retries - 1:
                    raise
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {retries} attempts")
    
    async def _make_private_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """Make a private API request with authentication."""
        if not self.api_key or not self.api_secret:
            raise Exception("API key and secret are required for private endpoints")
        
        # TODO: Implement Kraken authentication
        # This is a placeholder - actual implementation requires nonce and signature
        raise NotImplementedError("Kraken private API not yet implemented")
    
    async def fetch_historical_data(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Candle]:
        """Fetch historical OHLCV data from Kraken."""
        end = end or datetime.utcnow()
        cache_key = self._get_cache_key(pair, timeframe, start, end)
        
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded {len(cached_data)} candles from cache for {pair} {timeframe}")
                return cached_data
        
        # Kraken uses seconds since epoch
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        # Kraken has a limit of 720 candles per request
        interval_seconds = self._map_timeframe(timeframe) * 60
        max_candles = 720
        max_seconds = max_candles * interval_seconds
        
        all_candles = []
        current_start = start_ts
        
        try:
            while current_start < end_ts:
                # Calculate end time for this batch
                current_end = min(current_start + max_seconds, end_ts)
                
                response = await self._make_public_request(
                    'OHLC',
                    params={
                        'pair': pair,
                        'interval': self._map_timeframe(timeframe),
                        'since': current_start,
                    }
                )
                
                # Response format: {'result': {pair: [candles], 'last': last_timestamp}}
                pair_data = next(iter(response['result'].values()))
                
                for candle_data in pair_data:
                    if len(candle_data) < 8:  # OHLCV + count + vwap + trades
                        continue
                        
                    candle = Candle(
                        timestamp=pd.to_datetime(candle_data[0], unit='s'),
                        open=float(candle_data[1]),
                        high=float(candle_data[2]),
                        low=float(candle_data[3]),
                        close=float(candle_data[4]),
                        volume=float(candle_data[6])  # VWAP volume
                    )
                    all_candles.append(candle)
                
                # Update for next batch
                current_start = current_end
                
                # Check if we have all data
                if len(pair_data) < max_candles:
                    break
            
            # Save to cache
            if all_candles and use_cache:
                self._save_to_cache(cache_key, all_candles)
            
            return all_candles
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    async def fetch_latest_data(
        self,
        pair: str,
        timeframe: Timeframe,
        limit: int = 100
    ) -> List[Candle]:
        """Fetch the most recent candle data."""
        try:
            response = await self._make_public_request(
                'OHLC',
                params={
                    'pair': pair,
                    'interval': self._map_timeframe(timeframe),
                }
            )
            
            candles = []
            pair_data = next(iter(response['result'].values()))
            
            for candle_data in pair_data[-limit:]:  # Get last N candles
                if len(candle_data) < 8:
                    continue
                    
                candle = Candle(
                    timestamp=pd.to_datetime(candle_data[0], unit='s'),
                    open=float(candle_data[1]),
                    high=float(candle_data[2]),
                    low=float(candle_data[3]),
                    close=float(candle_data[4]),
                    volume=float(candle_data[6])
                )
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch latest data: {e}")
            raise
    
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """Get list of available trading pairs."""
        try:
            response = await self._make_public_request('AssetPairs')
            return list(response.get('result', {}).values())
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            response = await self._make_private_request('Balance')
            return response.get('result', {})
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise


# Factory function to create the appropriate connector
def create_connector(connector_type: str, **kwargs) -> BaseConnector:
    """Create a connector instance based on type."""
    connectors = {
        'oanda': OandaConnector,
        'kraken': KrakenConnector,
    }
    
    connector_class = connectors.get(connector_type.lower())
    if not connector_class:
        raise ValueError(f"Unknown connector type: {connector_type}")
    
    return connector_class(config=kwargs)
