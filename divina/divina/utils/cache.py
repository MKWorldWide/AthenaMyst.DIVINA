""
Caching utilities with TTL (time-to-live) support.
"""
import asyncio
import json
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar, Union, cast

import aiofiles
from loguru import logger

T = TypeVar('T')

class CacheEntry(Generic[T]):
    """A single cache entry with value and expiration time."""
    
    __slots__ = ('value', 'expires_at')
    
    def __init__(self, value: T, ttl_seconds: float):
        """Initialize a cache entry with a value and TTL in seconds."""
        self.value = value
        self.expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else float('inf')
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at


class MemoryCache(Generic[T]):
    """In-memory cache with TTL support and LRU eviction."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        evict_on_full: bool = True
    ):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            default_ttl: Default TTL in seconds for new entries
            evict_on_full: Whether to evict old entries when the cache is full
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.evict_on_full = evict_on_full
        self._cache: Dict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to default_ttl)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        
        async with self._lock:
            # Evict expired entries
            self._evict_expired()
            
            # Evict oldest entry if cache is full and eviction is enabled
            if len(self._cache) >= self.max_size and self.evict_on_full:
                # Remove the first item (oldest)
                self._cache.popitem(last=False)
            
            # Add the new entry
            self._cache[key] = CacheEntry(value, ttl)
            # Move to end to make it most recently used
            self._cache.move_to_end(key)
    
    async def get(self, key: str, default: Any = None) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key is not found or expired
            
        Returns:
            The cached value or default if not found or expired
        """
        async with self._lock:
            if key not in self._cache:
                return default
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                return default
            
            # Move to end to make it most recently used
            self._cache.move_to_end(key)
            return entry.value
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """Get the number of items in the cache (including expired but not yet evicted)."""
        async with self._lock:
            return len(self._cache)
    
    async def cleanup(self) -> None:
        """Clean up expired entries."""
        async with self._lock:
            self._evict_expired()
    
    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]


class DiskCache(Generic[T]):
    """Disk-based cache with TTL support."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: float = 3600.0,  # 1 hour
        max_size_mb: float = 100.0,   # 100 MB
        compress: bool = True
    ):
        """Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds for new entries
            max_size_mb: Maximum cache size in MB
            compress: Whether to compress stored data
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.compress = compress
        self._lock = asyncio.Lock()
        self._current_size = 0
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize current size
        self._update_size()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the filesystem path for a cache key."""
        # Simple hash to avoid filesystem issues with special characters
        key_hash = str(abs(hash(key)) % (10 ** 8)).zfill(8)
        return self.cache_dir / f"{key_hash}.cache"
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time to live in seconds (defaults to default_ttl)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else float('inf')
        
        # Serialize the value
        data = {
            'value': value,
            'expires_at': expires_at,
            'key': key,
            'created_at': time.time()
        }
        
        # Convert to JSON
        try:
            json_data = json.dumps(data).encode('utf-8')
        except (TypeError, OverflowError) as e:
            logger.error(f"Failed to serialize value for key {key}: {e}")
            return
        
        # Compress if enabled
        if self.compress:
            try:
                import zlib
                json_data = zlib.compress(json_data)
            except Exception as e:
                logger.warning(f"Failed to compress data for key {key}: {e}")
        
        # Check cache size and clean up if needed
        await self._enforce_size_limit()
        
        # Write to disk
        cache_path = self._get_cache_path(key)
        
        try:
            async with self._lock:
                async with aiofiles.open(cache_path, 'wb') as f:
                    await f.write(json_data)
                
                # Update size
                self._current_size += cache_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to write cache for key {key}: {e}")
    
    async def get(self, key: str, default: Any = None) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key is not found or expired
            
        Returns:
            The cached value or default if not found or expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return default
        
        try:
            async with self._lock:
                async with aiofiles.open(cache_path, 'rb') as f:
                    data = await f.read()
                
                # Decompress if needed
                if self.compress:
                    try:
                        import zlib
                        data = zlib.decompress(data)
                    except Exception as e:
                        logger.warning(f"Failed to decompress data for key {key}: {e}")
                        return default
                
                # Parse JSON
                try:
                    entry = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cache entry for key {key}: {e}")
                    return default
                
                # Check expiration
                if entry.get('expires_at', 0) < time.time():
                    await self.delete(key)
                    return default
                
                return cast(T, entry['value'])
                
        except Exception as e:
            logger.error(f"Failed to read cache for key {key}: {e}")
            return default
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return False
        
        try:
            async with self._lock:
                size = cache_path.stat().st_size
                cache_path.unlink()
                self._current_size = max(0, self._current_size - size)
                return True
        except Exception as e:
            logger.error(f"Failed to delete cache for key {key}: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        async with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")
            
            self._current_size = 0
    
    async def cleanup(self) -> None:
        """Clean up expired entries."""
        async with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    async with aiofiles.open(cache_file, 'rb') as f:
                        data = await f.read()
                    
                    if self.compress:
                        import zlib
                        data = zlib.decompress(data)
                    
                    entry = json.loads(data.decode('utf-8'))
                    
                    if entry.get('expires_at', 0) < time.time():
                        size = cache_file.stat().st_size
                        cache_file.unlink()
                        self._current_size = max(0, self._current_size - size)
                        
                except Exception as e:
                    logger.error(f"Failed to clean up cache file {cache_file}: {e}")
    
    async def size(self) -> int:
        """Get the number of items in the cache."""
        async with self._lock:
            return len(list(self.cache_dir.glob("*.cache")))
    
    def _update_size(self) -> None:
        """Update the current size of the cache."""
        self._current_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.cache")
            if f.is_file()
        )
    
    async def _enforce_size_limit(self) -> None:
        """Enforce the maximum cache size by removing oldest entries."""
        if self._current_size < self.max_size_bytes:
            return
        
        # Get all cache files sorted by last modified time (oldest first)
        cache_files = sorted(
            self.cache_dir.glob("*.cache"),
            key=lambda f: f.stat().st_mtime
        )
        
        async with self._lock:
            # Delete oldest files until we're under the size limit
            for cache_file in cache_files:
                if self._current_size < self.max_size_bytes * 0.9:  # Go down to 90% of max
                    break
                    
                try:
                    size = cache_file.stat().st_size
                    cache_file.unlink()
                    self._current_size = max(0, self._current_size - size)
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")


class HybridCache(Generic[T]):
    """Hybrid in-memory and disk cache with TTL support."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        memory_max_size: int = 1000,
        memory_ttl: float = 300.0,  # 5 minutes
        disk_ttl: float = 86400.0,   # 24 hours
        disk_max_size_mb: float = 100.0,  # 100 MB
        compress: bool = True
    ):
        """Initialize the hybrid cache.
        
        Args:
            cache_dir: Directory to store disk cache files
            memory_max_size: Maximum number of items in memory cache
            memory_ttl: TTL for memory cache entries in seconds
            disk_ttl: TTL for disk cache entries in seconds
            disk_max_size_mb: Maximum size of disk cache in MB
            compress: Whether to compress disk cache data
        """
        self.memory = MemoryCache[T](
            max_size=memory_max_size,
            default_ttl=memory_ttl
        )
        self.disk = DiskCache[T](
            cache_dir=cache_dir,
            default_ttl=disk_ttl,
            max_size_mb=disk_max_size_mb,
            compress=compress
        )
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        memory_only: bool = False
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to the respective cache's default)
            memory_only: If True, only store in memory cache
        """
        # Set in memory cache
        await self.memory.set(key, value, ttl)
        
        # Also set in disk cache unless memory_only is True
        if not memory_only:
            await self.disk.set(key, value, ttl)
    
    async def get(
        self,
        key: str,
        default: Any = None,
        check_disk: bool = True
    ) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key is not found or expired
            check_disk: Whether to check disk cache if not found in memory
            
        Returns:
            The cached value or default if not found or expired
        """
        # First try memory cache
        value = await self.memory.get(key, default=None)
        
        if value is not None and value != default:
            return value
        
        # If not in memory and we should check disk
        if check_disk:
            value = await self.disk.get(key, default=None)
            
            # If found in disk, update memory cache
            if value is not None and value != default:
                await self.memory.set(key, value)
                return value
        
        return default
    
    async def delete(self, key: str) -> bool:
        """Delete a key from both caches."""
        mem_result = await self.memory.delete(key)
        disk_result = await self.disk.delete(key)
        return mem_result or disk_result
    
    async def clear(self) -> None:
        """Clear both caches."""
        await asyncio.gather(
            self.memory.clear(),
            self.disk.clear()
        )
    
    async def cleanup(self) -> None:
        """Clean up expired entries in both caches."""
        await asyncio.gather(
            self.memory.cleanup(),
            self.disk.cleanup()
        )
    
    async def size(self) -> Dict[str, int]:
        """Get the number of items in each cache."""
        memory_size = await self.memory.size()
        disk_size = await self.disk.size()
        
        return {
            'memory': memory_size,
            'disk': disk_size,
            'total': memory_size + disk_size
        }
