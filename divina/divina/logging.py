""
Logging configuration for AthenaMyst:Divina.

Provides structured logging with correlation IDs and context.
"""
import sys
import logging
import json
from typing import Any, Dict, Optional
from uuid import uuid4

from loguru import logger
from loguru._defaults import LOGURU_FORMAT

from .config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
            
        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def serialize(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON."""
    subset = {
        "timestamp": record["time"].timestamp(),
        "level": record["level"].name,
        "message": record["message"],
        "name": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields if present
    if record.get("extra"):
        subset.update(record["extra"])
        
    return json.dumps(subset, default=str)


def setup_logging() -> None:
    """Configure logging for the application."""
    # Clear default logger
    logger.remove()
    
    # Configure console logging
    logger.add(
        sys.stderr,
        level=settings.log_level.value,
        format=LOGURU_FORMAT,
        serialize=settings.log_json,
        backtrace=True,
        diagnose=not settings.is_production,
        enqueue=True,  # Async logging
    )
    
    # Configure file logging in production
    if settings.is_production:
        logger.add(
            "logs/divina_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            level="INFO",
            enqueue=True,
            serialize=settings.log_json,
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Disable noisy loggers
    for name in ["uvicorn", "uvicorn.error", "fastapi"]:
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False


def configure_logging() -> None:
    ""
    Configure logging with context and correlation ID.
    
    Call this early in the application startup.
    """
    setup_logging()
    
    # Set correlation ID for the current context
    correlation_id = f"divina_{uuid4().hex[:8]}"
    logger.configure(extra={"correlation_id": correlation_id})
    
    logger.info("Logging configured", extra={"env": settings.env.value})
