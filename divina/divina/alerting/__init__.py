""
Alerting module for AthenaMyst:Divina.

This module provides idempotent alerting functionality with support for
multiple backends (Discord, Email, etc.) and deduplication.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, BaseSettings, Field, HttpUrl, validator

from ..state import State, state_context, state_property
from ..utils.cache import HybridCache
from ..utils.fp import ensure_utc, pure
from ..utils.time_utils import now_utc, parse_duration


class AlertLevel(str, Enum):
    """Alert severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertRecipient(BaseModel):
    """A recipient for alerts."""
    name: str
    contact: str
    enabled: bool = True
    alert_levels: List[AlertLevel] = Field(
        default_factory=lambda: [
            AlertLevel.INFO,
            AlertLevel.WARNING,
            AlertLevel.ERROR,
            AlertLevel.CRITICAL,
        ]
    )


class AlertContent(BaseModel):
    """Content of an alert."""
    title: str
    message: str
    level: AlertLevel = AlertLevel.INFO
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=now_utc)
    dedupe_key: Optional[str] = None
    ttl: Optional[timedelta] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            timedelta: lambda v: v.total_seconds() if v else None,
        }
    
    @validator('timestamp', pre=True, always=True)
    def ensure_utc_timestamp(cls, v):
        """Ensure the timestamp is timezone-aware and in UTC."""
        if v is None:
            return now_utc()
        return ensure_utc(v)
    
    @property
    def id(self) -> str:
        """Generate a unique ID for this alert based on its content."""
        if self.dedupe_key:
            return self.dedupe_key
        
        # Create a hash of the alert content for deduplication
        content = f"{self.title}:{self.message}:{self.level}"
        if self.data:
            content += f":{json.dumps(self.data, sort_keys=True)}"
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class AlertResult(BaseModel):
    """Result of sending an alert."""
    success: bool
    message: str
    alert_id: str
    backend: str
    timestamp: datetime = Field(default_factory=now_utc)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class AlertBackend(ABC):
    """Base class for alert backends."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the alert backend."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize the alert backend (implemented by subclasses)."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: AlertContent) -> AlertResult:
        """Send an alert (implemented by subclasses)."""
        pass
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class DiscordWebhookBackend(AlertBackend):
    """Alert backend for Discord webhooks."""
    
    def __init__(
        self,
        webhook_url: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="discord", **kwargs)
        self.webhook_url = webhook_url
        self.username = username or "AthenaMyst Alert"
        self.avatar_url = avatar_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _initialize(self) -> None:
        """Initialize the Discord webhook client."""
        self.session = aiohttp.ClientSession()
    
    async def send_alert(self, alert: AlertContent) -> AlertResult:
        """Send an alert to a Discord webhook."""
        if not self.enabled:
            return AlertResult(
                success=False,
                message="Backend disabled",
                alert_id=alert.id,
                backend=self.name
            )
        
        if not self.session:
            await self.initialize()
        
        # Map alert levels to Discord colors
        color_map = {
            AlertLevel.DEBUG: 0x9E9E9E,    # Grey
            AlertLevel.INFO: 0x2196F3,     # Blue
            AlertLevel.WARNING: 0xFFC107,  # Amber
            AlertLevel.ERROR: 0xF44336,    # Red
            AlertLevel.CRITICAL: 0x9C27B0, # Purple
        }
        
        # Create embed
        embed = {
            "title": f"{alert.level}: {alert.title}",
            "description": alert.message,
            "color": color_map.get(alert.level, 0x000000),
            "timestamp": alert.timestamp.isoformat(),
        }
        
        # Add fields from data
        if alert.data:
            embed["fields"] = [
                {
                    "name": str(key),
                    "value": str(value)[:1024],  # Discord has a 1024 char limit per field
                    "inline": True,
                }
                for key, value in alert.data.items()
            ][:25]  # Max 25 fields per embed
        
        # Prepare payload
        payload = {
            "username": self.username,
            "embeds": [embed],
        }
        
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url
        
        try:
            async with self.session.post(  # type: ignore
                self.webhook_url,
                json=payload,
                timeout=10.0
            ) as response:
                if response.status in (200, 204):
                    return AlertResult(
                        success=True,
                        message="Alert sent successfully",
                        alert_id=alert.id,
                        backend=self.name
                    )
                
                error_text = await response.text()
                return AlertResult(
                    success=False,
                    message=f"Failed to send alert: {response.status} {error_text}",
                    alert_id=alert.id,
                    backend=self.name
                )
        except Exception as e:
            return AlertResult(
                success=False,
                message=f"Error sending alert: {str(e)}",
                alert_id=alert.id,
                backend=self.name
            )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class ConsoleBackend(AlertBackend):
    """Alert backend that prints to console (for testing/debugging)."""
    
    async def send_alert(self, alert: AlertContent) -> AlertResult:
        """Print the alert to the console."""
        if not self.enabled:
            return AlertResult(
                success=False,
                message="Backend disabled",
                alert_id=alert.id,
                backend=self.name
            )
        
        print(f"\n=== {alert.level}: {alert.title} ===")
        print(f"Time: {alert.timestamp.isoformat()}")
        print(f"Message: {alert.message}")
        
        if alert.data:
            print("Data:")
            for key, value in alert.data.items():
                print(f"  {key}: {value}")
        
        print("=" * 40 + "\n")
        
        return AlertResult(
            success=True,
            message="Alert printed to console",
            alert_id=alert.id,
            backend=self.name
        )


class AlertManagerSettings(BaseSettings):
    """Settings for the AlertManager."""
    
    class Config:
        env_prefix = "ALERT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    # Default TTL for alert deduplication (in seconds)
    default_ttl: int = 3600  # 1 hour
    
    # Backend configurations
    backends: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "console": {
                "enabled": True,
                "class": "divina.alerting.ConsoleBackend",
            },
        }
    )
    
    # Alert level mappings
    min_alert_level: AlertLevel = AlertLevel.INFO
    
    # Rate limiting
    rate_limit: int = 60  # Max alerts per minute
    rate_limit_window: int = 60  # Window in seconds
    
    # Alert history
    max_history: int = 1000  # Max number of alerts to keep in history
    
    @validator('backends', pre=True)
    def parse_backends(cls, v):
        """Parse backend configurations."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v or {}


class AlertManager:
    """Manages alerting with support for multiple backends and deduplication."""
    
    def __init__(self, settings: Optional[AlertManagerSettings] = None):
        """Initialize the AlertManager."""
        self.settings = settings or AlertManagerSettings()
        self.backends: Dict[str, AlertBackend] = {}
        self._initialized = False
        self._rate_limit_count = 0
        self._rate_limit_reset = now_utc()
        self._lock = asyncio.Lock()
        
        # Initialize cache for deduplication
        self._cache = HybridCache(
            cache_dir=os.path.expanduser("~/.cache/athenamyst/alerts"),
            memory_max_size=1000,
            memory_ttl=self.settings.default_ttl,
            disk_ttl=self.settings.default_ttl * 2,
            disk_max_size_mb=10,
        )
        
        # Initialize state
        self._state = State()
        self._state["alert_history"] = []
    
    async def initialize(self) -> None:
        """Initialize the alert manager and all backends."""
        if self._initialized:
            return
        
        # Initialize backends
        for name, config in self.settings.backends.items():
            if not config.get("enabled", True):
                continue
            
            try:
                # Import the backend class
                module_path, class_name = config["class"].rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                backend_class = getattr(module, class_name)
                
                # Create the backend instance
                backend_config = {k: v for k, v in config.items() if k != "class"}
                backend = backend_class(name=name, **backend_config)
                
                # Initialize the backend
                await backend.initialize()
                
                # Add to backends
                self.backends[name] = backend
                logger.info(f"Initialized alert backend: {name} ({backend_class.__name__})")
                
            except Exception as e:
                logger.error(f"Failed to initialize alert backend {name}: {e}")
        
        self._initialized = True
        logger.info(f"AlertManager initialized with {len(self.backends)} backends")
    
    async def send_alert(
        self,
        title: str,
        message: str,
        level: Union[AlertLevel, str] = AlertLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        dedupe_key: Optional[str] = None,
        ttl: Optional[Union[int, float, str, timedelta]] = None,
        backends: Optional[List[str]] = None,
    ) -> List[AlertResult]:
        """Send an alert to all configured backends.
        
        Args:
            title: Alert title/summary.
            message: Detailed alert message.
            level: Alert severity level.
            data: Additional data to include with the alert.
            dedupe_key: Optional key for deduplication.
            ttl: Time-to-live for deduplication (defaults to settings.default_ttl).
            backends: List of backend names to send to (None for all).
            
        Returns:
            List of alert results, one per backend.
        """
        if not self._initialized:
            await self.initialize()
        
        # Parse TTL
        if ttl is None:
            ttl_seconds = self.settings.default_ttl
        elif isinstance(ttl, (int, float)):
            ttl_seconds = float(ttl)
        elif isinstance(ttl, str):
            ttl_seconds = parse_duration(ttl).total_seconds()
        elif isinstance(ttl, timedelta):
            ttl_seconds = ttl.total_seconds()
        else:
            ttl_seconds = self.settings.default_ttl
        
        # Create alert
        alert = AlertContent(
            title=title,
            message=message,
            level=AlertLevel(level.upper() if isinstance(level, str) else level),
            data=data or {},
            dedupe_key=dedupe_key,
            ttl=timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None,
        )
        
        # Check rate limiting
        async with self._lock:
            now = now_utc()
            
            # Reset rate limit counter if window has passed
            if now >= self._rate_limit_reset:
                self._rate_limit_count = 0
                self._rate_limit_reset = now + timedelta(
                    seconds=self.settings.rate_limit_window
                )
            
            # Check if we're over the rate limit
            if self._rate_limit_count >= self.settings.rate_limit:
                logger.warning(
                    f"Rate limit exceeded ({self._rate_limit_count} alerts in last "
                    f"{self.settings.rate_limit_window}s), alert suppressed: {alert.title}"
                )
                return [
                    AlertResult(
                        success=False,
                        message=f"Rate limit exceeded: {self._rate_limit_count} alerts in last {self.settings.rate_limit_window}s",
                        alert_id=alert.id,
                        backend="all",
                    )
                ]
            
            # Increment rate limit counter
            self._rate_limit_count += 1
        
        # Check alert level
        if AlertLevel(alert.level) < AlertLevel(self.settings.min_alert_level):
            logger.debug(f"Alert level {alert.level} below minimum {self.settings.min_alert_level}, suppressing")
            return [
                AlertResult(
                    success=True,
                    message=f"Alert level {alert.level} below minimum {self.settings.min_alert_level}, suppressed",
                    alert_id=alert.id,
                    backend="all",
                )
            ]
        
        # Check for duplicates
        cache_key = f"alert:{alert.id}"
        if await self._cache.get(cache_key) is not None:
            logger.debug(f"Duplicate alert detected, suppressing: {alert.title} ({alert.id})")
            return [
                AlertResult(
                    success=True,
                    message="Duplicate alert suppressed",
                    alert_id=alert.id,
                    backend="all",
                )
            ]
        
        # Cache the alert ID to prevent duplicates
        await self._cache.set(
            cache_key,
            {"title": alert.title, "timestamp": alert.timestamp.isoformat()},
            ttl=ttl_seconds,
        )
        
        # Determine which backends to use
        backend_names = backends or list(self.backends.keys())
        if not backend_names:
            logger.warning("No alert backends configured or enabled")
            return []
        
        # Send to all specified backends in parallel
        tasks = []
        for name in backend_names:
            if name not in self.backends:
                logger.warning(f"Alert backend not found: {name}")
                continue
            
            tasks.append(self._send_to_backend(name, alert))
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error sending alert: {result}")
                continue
            valid_results.append(result)
            
            # Log failures
            if not result.success:
                logger.error(
                    f"Failed to send alert to {result.backend}: {result.message}"
                )
        
        # Add to history
        await self._add_to_history(alert, valid_results)
        
        return valid_results
    
    async def _send_to_backend(
        self,
        backend_name: str,
        alert: AlertContent,
    ) -> AlertResult:
        """Send an alert to a specific backend."""
        backend = self.backends[backend_name]
        
        try:
            result = await backend.send_alert(alert)
            return result
        except Exception as e:
            logger.exception(f"Error sending alert to {backend_name}")
            return AlertResult(
                success=False,
                message=f"Error: {str(e)}",
                alert_id=alert.id,
                backend=backend_name,
            )
    
    async def _add_to_history(
        self,
        alert: AlertContent,
        results: List[AlertResult],
    ) -> None:
        """Add an alert to the history."""
        # Clean up old history if needed
        history = self._state["alert_history"]
        if len(history) >= self.settings.max_history:
            # Remove the oldest 10% of history when we hit the limit
            remove_count = max(10, self.settings.max_history // 10)
            history = history[remove_count:]
        
        # Add the new alert
        history.append({
            "alert": alert.dict(),
            "results": [r.dict() for r in results],
            "timestamp": now_utc().isoformat(),
        })
        
        # Update state
        self._state["alert_history"] = history
    
    async def get_history(
        self,
        limit: int = 100,
        level: Optional[Union[AlertLevel, str]] = None,
        backend: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        history = self._state.get("alert_history", [])
        
        # Apply filters
        filtered = []
        for entry in reversed(history):
            if level is not None:
                alert_level = entry["alert"].get("level")
                if isinstance(level, str):
                    level = AlertLevel(level.upper())
                if AlertLevel(alert_level) < level:
                    continue
            
            if backend is not None:
                results = entry.get("results", [])
                if not any(r.get("backend") == backend for r in results):
                    continue
            
            filtered.append(entry)
            if len(filtered) >= limit:
                break
        
        return filtered
    
    async def close(self) -> None:
        """Clean up resources."""
        # Close all backends
        for backend in self.backends.values():
            try:
                await backend.close()
            except Exception as e:
                logger.error(f"Error closing alert backend {backend.name}: {e}")
        
        # Clear state
        self.backends.clear()
        self._initialized = False


# Global instance for convenience
alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the global AlertManager instance."""
    global alert_manager
    
    if alert_manager is None:
        alert_manager = AlertManager()
    
    return alert_manager


async def send_alert(
    title: str,
    message: str,
    level: Union[AlertLevel, str] = AlertLevel.INFO,
    **kwargs
) -> List[AlertResult]:
    """Send an alert using the global AlertManager."""
    manager = get_alert_manager()
    return await manager.send_alert(title, message, level, **kwargs)


async def init_alert_manager(settings: Optional[AlertManagerSettings] = None) -> AlertManager:
    """Initialize the global AlertManager."""
    global alert_manager
    
    if alert_manager is not None:
        await alert_manager.close()
    
    alert_manager = AlertManager(settings)
    await alert_manager.initialize()
    
    return alert_manager


async def close_alert_manager() -> None:
    """Close the global AlertManager."""
    global alert_manager
    
    if alert_manager is not None:
        await alert_manager.close()
        alert_manager = None
