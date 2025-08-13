""
Structured logging with correlation IDs for AthenaMyst:Divina.

This module provides a structured logging setup with correlation IDs for
traceability across services and components.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler, SysLogHandler
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import loguru
from loguru import logger
from pydantic import BaseModel, Field, validator

from ..utils.time_utils import now_utc

# Type variable for generic function wrapping
T = TypeVar('T', bound=Callable[..., Any])

# Context variable for correlation ID
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)

# Context variable for additional log context
log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'log_context', default_factory=dict
)

# Request ID for FastAPI request tracking
request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)

# User context for authentication
user_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'user_context', default_factory=dict
)


class LogLevel(str, Enum):
    """Standard log levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogConfig(BaseModel):
    """Logging configuration."""
    
    class Config:
        env_prefix = "LOG_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    # General settings
    level: LogLevel = Field(
        LogLevel.INFO,
        description="Minimum log level to output"
    )
    json_format: bool = Field(
        True,
        description="Whether to output logs in JSON format"
    )
    colorize: bool = Field(
        not bool(os.getenv("NO_COLOR", "")),
        description="Enable colored output"
    )
    backtrace: bool = Field(
        True,
        description="Enable exception trace in logs"
    )
    diagnose: bool = Field(
        os.getenv("ENV", "production").lower() != "production",
        description="Enable exception diagnosis (can leak sensitive data)"
    )
    
    # File logging
    file_path: Optional[str] = Field(
        None,
        description="Path to log file (if None, logs to stderr)"
    )
    file_rotation: str = Field(
        "100 MB",
        description="Log rotation size or time (e.g., '100 MB', '1 day')"
    )
    file_retention: str = Field(
        "30 days",
        description="How long to keep rotated logs (e.g., '30 days')"
    )
    file_compression: str = Field(
        "gz",
        description="Compression format for rotated logs ('gz', 'zip', or '' for none)"
    )
    
    # Syslog settings
    syslog_enabled: bool = Field(
        False,
        description="Enable syslog logging"
    )
    syslog_address: str = Field(
        "/dev/log",
        description="Syslog server address (path or host:port)"
    )
    syslog_facility: str = Field(
        "user",
        description="Syslog facility (user, local0, etc.)"
    )
    
    # Context settings
    include_correlation_id: bool = Field(
        True,
        description="Include correlation ID in logs"
    )
    include_request_id: bool = Field(
        True,
        description="Include request ID in logs"
    )
    include_user_context: bool = Field(
        True,
        description="Include user context in logs"
    )
    include_extra: bool = Field(
        True,
        description="Include extra context in logs"
    )
    
    # Performance tracking
    slow_threshold: float = Field(
        1.0,
        description="Log warnings for slow operations (seconds)"
    )
    
    @validator('level', pre=True)
    def validate_level(cls, v):
        """Ensure log level is uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return request_id.get()


def get_log_context() -> Dict[str, Any]:
    """Get the current log context."""
    return log_context.get()


def get_user_context() -> Dict[str, Any]:
    """Get the current user context."""
    return user_context.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set the correlation ID.
    
    Args:
        cid: Optional correlation ID. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def set_request_id(rid: Optional[str] = None) -> str:
    """Set the request ID.
    
    Args:
        rid: Optional request ID. If None, generates a new UUID.
        
    Returns:
        The request ID that was set.
    """
    if rid is None:
        rid = str(uuid.uuid4())
    request_id.set(rid)
    return rid


def update_log_context(**kwargs) -> Dict[str, Any]:
    """Update the log context with new key-value pairs.
    
    Args:
        **kwargs: Key-value pairs to add to the log context.
        
    Returns:
        The updated log context.
    """
    ctx = log_context.get().copy()
    ctx.update(kwargs)
    log_context.set(ctx)
    return ctx


def update_user_context(**kwargs) -> Dict[str, Any]:
    """Update the user context with new key-value pairs.
    
    Args:
        **kwargs: Key-value pairs to add to the user context.
        
    Returns:
        The updated user context.
    """
    ctx = user_context.get().copy()
    ctx.update(kwargs)
    user_context.set(ctx)
    return ctx


def clear_log_context() -> None:
    """Clear the log context."""
    log_context.set({})


def clear_user_context() -> None:
    """Clear the user context."""
    user_context.set({})


@contextmanager
def log_context_scope(**kwargs):
    """Context manager for scoped log context."""
    old_ctx = log_context.get().copy()
    update_log_context(**kwargs)
    try:
        yield
    finally:
        log_context.set(old_ctx)


@asynccontextmanager
async def async_log_context_scope(**kwargs):
    """Async context manager for scoped log context."""
    old_ctx = log_context.get().copy()
    update_log_context(**kwargs)
    try:
        yield
    finally:
        log_context.set(old_ctx)


def with_correlation_id(cid: Optional[str] = None):
    """Decorator to set correlation ID for a function."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                old_cid = get_correlation_id()
                set_correlation_id(cid)
                try:
                    return await func(*args, **kwargs)
                finally:
                    correlation_id.set(old_cid)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                old_cid = get_correlation_id()
                set_correlation_id(cid)
                try:
                    return func(*args, **kwargs)
                finally:
                    correlation_id.set(old_cid)
            return sync_wrapper
    return decorator


def with_request_id(rid: Optional[str] = None):
    """Decorator to set request ID for a function."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                old_rid = get_request_id()
                set_request_id(rid)
                try:
                    return await func(*args, **kwargs)
                finally:
                    request_id.set(old_rid)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                old_rid = get_request_id()
                set_request_id(rid)
                try:
                    return func(*args, **kwargs)
                finally:
                    request_id.set(old_rid)
            return sync_wrapper
    return decorator


def log_execution_time(level: str = "DEBUG", message: str = "{function} completed in {time:.3f}s"):
    """Decorator to log function execution time."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.monotonic() - start_time
                    logger.log(
                        level,
                        message,
                        function=f"{func.__module__}.{func.__name__}",
                        time=duration,
                    )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.monotonic() - start_time
                    logger.log(
                        level,
                        message,
                        function=f"{func.__module__}.{func.__name__}",
                        time=duration,
                    )
            return sync_wrapper
    return decorator


def log_exceptions(level: str = "ERROR", reraise: bool = True):
    """Decorator to log exceptions with context."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.opt(exception=e).log(
                        level,
                        f"Exception in {func.__module__}.{func.__name__}: {e}",
                        exc_info=True,
                    )
                    if reraise:
                        raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.opt(exception=e).log(
                        level,
                        f"Exception in {func.__module__}.{func.__name__}: {e}",
                        exc_info=True,
                    )
                    if reraise:
                        raise
            return sync_wrapper
    return decorator


def configure_logging(config: Optional[LogConfig] = None) -> None:
    """Configure logging with the given configuration.
    
    Args:
        config: Logging configuration. If None, uses default config.
    """
    if config is None:
        config = LogConfig()
    
    # Remove default handler
    logger.remove()
    
    # Configure log format
    if config.json_format:
        def formatter(record: loguru.Record) -> str:
            # Extract log record fields
            record_dict = {
                "timestamp": datetime.fromtimestamp(
                    record["time"].timestamp(), tz=timezone.utc
                ).isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
            }
            
            # Add correlation ID if available
            cid = get_correlation_id()
            if cid and config.include_correlation_id:
                record_dict["correlation_id"] = cid
            
            # Add request ID if available
            rid = get_request_id()
            if rid and config.include_request_id:
                record_dict["request_id"] = rid
            
            # Add user context if available
            user_ctx = get_user_context()
            if user_ctx and config.include_user_context:
                record_dict["user"] = user_ctx
            
            # Add log context if available
            log_ctx = get_log_context()
            if log_ctx and config.include_extra:
                record_dict.update(log_ctx)
            
            # Add exception info if available
            if record["exception"] is not None:
                record_dict["exception"] = {
                    "type": record["exception"].type.__name__,
                    "message": str(record["exception"].value),
                    "traceback": "".join(traceback.format_tb(record["exception"].traceback)),
                }
            
            # Add any extra fields
            extra = {k: v for k, v in record["extra"].items() if not k.startswith("_")}
            if extra and config.include_extra:
                record_dict["extra"] = extra
            
            return json.dumps(record_dict, default=str) + "\n"
    else:
        # Simple text format for console
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        if config.include_correlation_id:
            format_str = "<magenta>{extra[correlation_id]}</magenta> | " + format_str
        
        if config.include_request_id:
            format_str = "<yellow>{extra[request_id]}</yellow> | " + format_str
        
        def formatter(record: loguru.Record) -> str:
            # Add correlation ID to record
            cid = get_correlation_id()
            if cid and config.include_correlation_id:
                record["extra"]["correlation_id"] = cid
            
            # Add request ID to record
            rid = get_request_id()
            if rid and config.include_request_id:
                record["extra"]["request_id"] = rid
            
            return format_str + "\n"
    
    # Configure console handler
    logger.add(
        sys.stderr,
        level=config.level,
        format=formatter,
        colorize=config.colorize,
        backtrace=config.backtrace,
        diagnose=config.diagnose,
    )
    
    # Configure file handler if path is provided
    if config.file_path:
        logger.add(
            config.file_path,
            rotation=config.file_rotation,
            retention=config.file_retention,
            compression=config.file_compression or None,
            level=config.level,
            format=formatter,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            enqueue=True,  # Use thread-safe queue
        )
    
    # Configure syslog if enabled
    if config.syslog_enabled:
        try:
            syslog_handler = SysLogHandler(address=config.syslog_address)
            syslog_handler.setFormatter(
                logging.Formatter(
                    f"%(asctime)s %(name)s[%(process)d]: %(message)s",
                    datefmt="%b %d %H:%M:%S",
                )
            )
            
            class SyslogLogger:
                def write(self, message):
                    if message.strip():
                        syslog_handler.emit(
                            logging.LogRecord(
                                name="athenamyst",
                                level=logging.INFO,
                                pathname="",
                                lineno=0,
                                msg=message,
                                args=(),
                                exc_info=None,
                            )
                        )
                
                def flush(self):
                    pass
            
            logger.add(
                SyslogLogger(),
                level=config.level,
                format=formatter,
                backtrace=config.backtrace,
                diagnose=config.diagnose,
            )
        except Exception as e:
            logger.error(f"Failed to configure syslog: {e}")
    
    # Configure root logger to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set log levels for noisy libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger.info("Logging configured", config=config.dict())


class InterceptHandler(logging.Handler):
    """Intercept standard logging and route to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where the logged message originated
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        
        # Log the message with context
        logger.opt(
            depth=depth,
            exception=record.exc_info,
            lazy=True,
            colors=True,
            record=True,
        ).log(
            level,
            record.getMessage(),
        )


# Initialize logging with default config
configure_logging()

# Export common log functions for convenience
info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception
trace = logger.trace
success = logger.success
log = logger.log

# Export context managers and decorators
__all__ = [
    # Logging functions
    "info",
    "debug",
    "warning",
    "error",
    "critical",
    "exception",
    "trace",
    "success",
    "log",
    
    # Context and IDs
    "get_correlation_id",
    "get_request_id",
    "get_log_context",
    "get_user_context",
    "set_correlation_id",
    "set_request_id",
    "update_log_context",
    "update_user_context",
    "clear_log_context",
    "clear_user_context",
    "log_context_scope",
    "async_log_context_scope",
    
    # Decorators
    "with_correlation_id",
    "with_request_id",
    "log_execution_time",
    "log_exceptions",
    
    # Configuration
    "LogConfig",
    "LogLevel",
    "configure_logging",
    
    # Types
    "T",
]
