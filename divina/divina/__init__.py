"""
AthenaMyst:Divina - Multi-timeframe FX Signal Engine

A high-performance, production-ready trading signal engine supporting
multi-timeframe analysis with Ichimoku, RSI, VWAP, and Volume indicators.
"""

__version__ = "0.1.0"

from .config import settings
from .logging import configure_logging

# Configure logging when the package is imported
configure_logging()
