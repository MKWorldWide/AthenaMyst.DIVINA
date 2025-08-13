""
Time-related utilities for consistent timezone handling.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Union, Any, Dict, List, Tuple
import time
import pytz

from loguru import logger

# Type aliases
Timestamp = Union[float, int, str, datetime]
Timezone = Union[str, timezone, pytz.BaseTzInfo]

def ensure_utc(dt: Optional[Timestamp] = None) -> datetime:
    """Ensure a datetime object is timezone-aware and in UTC.
    
    Args:
        dt: Input datetime, timestamp, or ISO format string. If None, uses current time.
        
    Returns:
        Timezone-aware datetime in UTC.
        
    Raises:
        ValueError: If the input cannot be converted to a datetime.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    
    # Handle string input
    if isinstance(dt, str):
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid datetime string: {dt}")
    
    # Handle numeric timestamp
    elif isinstance(dt, (int, float)):
        # Check if timestamp is in milliseconds
        if dt > 1e12:  # Roughly Sep 2001 in seconds
            dt = dt / 1000.0
        dt = datetime.fromtimestamp(dt, timezone.utc)
    
    # Ensure we have a datetime object
    if not isinstance(dt, datetime):
        raise ValueError(f"Cannot convert {type(dt).__name__} to datetime")
    
    # Ensure timezone-aware and in UTC
    if dt.tzinfo is None:
        logger.warning(f"Naive datetime provided, assuming UTC: {dt}")
        return dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(timezone.utc)

def to_utc_timestamp(dt: Optional[Timestamp] = None) -> float:
    """Convert a datetime or timestamp to UTC timestamp (seconds since epoch).
    
    Args:
        dt: Input datetime, timestamp, or ISO format string. If None, uses current time.
        
    Returns:
        Timestamp in seconds since epoch (UTC).
    """
    return ensure_utc(dt).timestamp()

def parse_timezone(tz: Optional[Timezone] = None) -> timezone:
    """Parse a timezone string or tzinfo object to a timezone.
    
    Args:
        tz: Timezone as string (e.g., 'UTC', 'America/New_York'), tzinfo, or None for UTC.
        
    Returns:
        timezone object.
        
    Raises:
        ValueError: If the timezone is not recognized.
    """
    if tz is None:
        return timezone.utc
    
    if isinstance(tz, timezone):
        return tz
    
    if isinstance(tz, str):
        if tz.upper() == 'UTC':
            return timezone.utc
        
        try:
            # Try as IANA timezone name
            return pytz.timezone(tz)
        except pytz.exceptions.UnknownTimeZoneError:
            # Try as offset like +HH:MM or -HH:MM
            try:
                if tz.startswith(('+', '-')) and ':' in tz:
                    hours, minutes = map(int, tz[1:].split(':'))
                    offset = hours * 3600 + minutes * 60
                    if tz.startswith('-'):
                        offset = -offset
                    return timezone(timedelta(seconds=offset))
            except (ValueError, IndexError):
                pass
            
            raise ValueError(f"Unknown timezone: {tz}")
    
    if hasattr(tz, 'utcoffset'):  # Check if it's a tzinfo-like object
        return tz
    
    raise ValueError(f"Invalid timezone: {tz}")

def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(timezone.utc)

def format_iso(dt: Optional[Timestamp] = None, with_tz: bool = True) -> str:
    """Format a datetime as ISO 8601 string.
    
    Args:
        dt: Input datetime, timestamp, or ISO format string. If None, uses current time.
        with_tz: Whether to include timezone information.
        
    Returns:
        ISO 8601 formatted string.
    """
    dt_utc = ensure_utc(dt)
    
    if with_tz:
        return dt_utc.isoformat()
    
    # Return in UTC without timezone indicator (Z)
    return dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f') if dt_utc.microsecond else dt_utc.strftime('%Y-%m-%dT%H:%M:%S')

def parse_duration(duration: Union[int, float, str, timedelta]) -> timedelta:
    """Parse a duration string or number into a timedelta.
    
    Supported formats:
    - '1d' or '1D' = 1 day
    - '1h' or '1H' = 1 hour
    - '1m' or '1M' = 1 minute
    - '1s' or '1S' = 1 second
    - '1d 2h 3m 4s' = 1 day, 2 hours, 3 minutes, 4 seconds
    - 3600 (int/float) = 3600 seconds
    
    Args:
        duration: Duration as string, number (seconds), or timedelta.
        
    Returns:
        timedelta object.
        
    Raises:
        ValueError: If the duration string is invalid.
    """
    if isinstance(duration, timedelta):
        return duration
    
    if isinstance(duration, (int, float)):
        return timedelta(seconds=duration)
    
    if not isinstance(duration, str):
        raise ValueError(f"Invalid duration type: {type(duration).__name__}")
    
    # Remove any whitespace and convert to lowercase
    duration = duration.strip().lower()
    
    # Check for simple format (e.g., '1d', '2h', etc.)
    if len(duration) > 1 and duration[-1] in ('d', 'h', 'm', 's'):
        try:
            value = float(duration[:-1])
            unit = duration[-1]
            
            if unit == 'd':
                return timedelta(days=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'm':
                return timedelta(minutes=value)
            elif unit == 's':
                return timedelta(seconds=value)
        except ValueError:
            pass
    
    # Check for compound format (e.g., '1d 2h 3m 4s')
    parts = duration.split()
    if len(parts) > 1:
        total = timedelta()
        for part in parts:
            total += parse_duration(part)
        return total
    
    # Try to parse as ISO 8601 duration (basic support)
    if duration.startswith('P'):
        try:
            # Simple implementation for common cases
            # For full ISO 8601, consider using isodate or similar library
            if 'T' in duration:
                date_part, time_part = duration[1:].split('T', 1)
            else:
                date_part, time_part = duration[1:], ''
            
            days = 0
            hours = 0
            minutes = 0
            seconds = 0
            
            # Parse date part
            if date_part:
                if 'D' in date_part:
                    days = int(date_part.split('D')[0])
            
            # Parse time part
            if 'H' in time_part:
                hours = int(time_part.split('H')[0])
                time_part = time_part.split('H', 1)[1]
            
            if 'M' in time_part:
                minutes = int(time_part.split('M')[0])
                time_part = time_part.split('M', 1)[1]
            
            if 'S' in time_part:
                seconds = float(time_part.split('S')[0])
            
            return timedelta(
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid ISO 8601 duration: {duration}") from e
    
    # Try to parse as a number of seconds
    try:
        return timedelta(seconds=float(duration))
    except ValueError as e:
        raise ValueError(f"Invalid duration format: {duration}") from e

def time_elapsed(start_time: Timestamp) -> float:
    """Calculate time elapsed since start_time in seconds.
    
    Args:
        start_time: Start time as datetime, timestamp, or ISO format string.
        
    Returns:
        Time elapsed in seconds.
    """
    return (now_utc() - ensure_utc(start_time)).total_seconds()

def time_until(target_time: Timestamp) -> float:
    """Calculate time remaining until target_time in seconds.
    
    Args:
        target_time: Target time as datetime, timestamp, or ISO format string.
        
    Returns:
        Time remaining in seconds (negative if target_time is in the past).
    """
    return (ensure_utc(target_time) - now_utc()).total_seconds()

def is_within_hours(
    dt: Timestamp,
    start_hour: int,
    end_hour: int,
    tz: Timezone = 'UTC'
) -> bool:
    """Check if a datetime is within specified hours of the day.
    
    Args:
        dt: Datetime to check.
        start_hour: Start hour (0-23).
        end_hour: End hour (0-23).
        tz: Timezone to use for hour comparison.
        
    Returns:
        True if the time is within the specified hours, False otherwise.
    """
    dt = ensure_utc(dt)
    tz_obj = parse_timezone(tz)
    
    # Convert to target timezone
    dt_local = dt.astimezone(tz_obj)
    current_hour = dt_local.hour
    
    # Handle overnight range (e.g., 22:00-06:00)
    if start_hour > end_hour:
        return current_hour >= start_hour or current_hour < end_hour
    
    # Handle same-day range (e.g., 09:00-17:00)
    return start_hour <= current_hour < end_hour

def next_occurrence(
    hour: int,
    minute: int = 0,
    second: int = 0,
    tz: Timezone = 'UTC'
) -> datetime:
    """Get the next occurrence of a specific time.
    
    Args:
        hour: Hour (0-23).
        minute: Minute (0-59).
        second: Second (0-59).
        tz: Timezone to use.
        
    Returns:
        The next occurrence of the specified time.
    """
    tz_obj = parse_timezone(tz)
    now = datetime.now(tz_obj)
    
    # Today's occurrence
    today = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    
    # If today's occurrence is in the future, return it
    if today > now:
        return today.astimezone(timezone.utc)
    
    # Otherwise, return tomorrow's occurrence
    return (today + timedelta(days=1)).astimezone(timezone.utc)

# Common time constants
ONE_SECOND = timedelta(seconds=1)
ONE_MINUTE = timedelta(minutes=1)
FIVE_MINUTES = timedelta(minutes=5)
FIFTEEN_MINUTES = timedelta(minutes=15)
THIRTY_MINUTES = timedelta(minutes=30)
ONE_HOUR = timedelta(hours=1)
SIX_HOURS = timedelta(hours=6)
TWELVE_HOURS = timedelta(hours=12)
ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(weeks=1)
ONE_MONTH = timedelta(days=30)  # Approximate
ONE_YEAR = timedelta(days=365)  # Approximate

# Common time formats
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
ISO_FORMAT_NO_MS = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
DATETIME_FORMAT = f"{DATE_FORMAT} {TIME_FORMAT}"

# Timezone constants
UTC = timezone.utc
