""
AthenaMyst:Divina - Main application package.

This package contains the core application logic and API endpoints.
"""
from pathlib import Path

# Ensure the app directory is treated as a Python package
__all__ = ["api", "main", "metrics"]

# Create logs directory if it doesn't exist
(Path(__file__).parent.parent / "logs").mkdir(exist_ok=True)
