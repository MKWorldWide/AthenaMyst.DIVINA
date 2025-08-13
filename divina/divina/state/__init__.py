""
State management for AthenaMyst:Divina.

This module provides a clean, type-safe way to manage application state
without relying on global variables. It uses a combination of dependency
injection and context managers to ensure thread safety and proper cleanup.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union,
    cast, get_args, get_origin, get_type_hints
)

from loguru import logger
from pydantic import BaseModel, ValidationError

from ..utils.fp import ensure_utc
from ..utils.time_utils import now_utc

# Type variables for generic state management
T = TypeVar('T')
M = TypeVar('M', bound=BaseModel)

# Context variable to track the current state
_current_state: ContextVar[Optional[State]] = ContextVar('_current_state', default=None)


class StateError(Exception):
    """Base exception for state-related errors."""
    pass


class StateNotInitializedError(StateError):
    """Raised when trying to access state before it's been initialized."""
    pass


class StateKeyError(StateError, KeyError):
    """Raised when a state key is not found."""
    pass


class StateValueError(StateError, ValueError):
    """Raised when a state value is invalid."""
    pass


class StateEntry(Generic[T]):
    """A single entry in the state with metadata and validation."""
    
    __slots__ = ('_value', '_default', '_validator', '_last_updated', '_ttl')
    
    def __init__(
        self,
        value: Optional[T] = None,
        default: Optional[Union[T, Callable[[], T]]] = None,
        validator: Optional[Callable[[T], bool]] = None,
        ttl: Optional[float] = None
    ):
        """Initialize a state entry.
        
        Args:
            value: Initial value (None means unset).
            default: Default value or factory function.
            validator: Optional validation function.
            ttl: Time-to-live in seconds (None means no expiration).
        """
        self._value: Optional[T] = value
        self._default = default
        self._validator = validator
        self._last_updated: Optional[datetime] = now_utc() if value is not None else None
        self._ttl = ttl
    
    @property
    def value(self) -> T:
        """Get the current value, falling back to default if unset."""
        if self._value is not None:
            # Check if the value has expired
            if self._ttl is not None and self._last_updated is not None:
                if (now_utc() - self._last_updated).total_seconds() > self._ttl:
                    self._value = None
                    self._last_updated = None
        
        if self._value is None and self._default is not None:
            if callable(self._default):
                self._value = self._default()
            else:
                self._value = self._default
            self._last_updated = now_utc()
        
        if self._value is None:
            raise StateKeyError("No value set and no default provided")
        
        return self._value
    
    @value.setter
    def value(self, new_value: T) -> None:
        """Set a new value with validation."""
        if new_value is None:
            self._value = None
            self._last_updated = None
            return
        
        # Validate the new value if a validator is provided
        if self._validator is not None and not self._validator(new_value):
            raise StateValueError(f"Invalid value: {new_value}")
        
        self._value = new_value
        self._last_updated = now_utc()
    
    @property
    def last_updated(self) -> Optional[datetime]:
        """Get when this value was last updated."""
        return self._last_updated
    
    @property
    def is_expired(self) -> bool:
        """Check if this value has expired."""
        if self._ttl is None or self._last_updated is None:
            return False
        return (now_utc() - self._last_updated).total_seconds() > self._ttl
    
    def get(self, default: Optional[T] = None) -> Optional[T]:
        """Get the value or a default if unset or expired."""
        try:
            return self.value
        except StateKeyError:
            return default
    
    def clear(self) -> None:
        """Clear the current value."""
        self._value = None
        self._last_updated = None


class State:
    """Thread-safe state container with type checking and validation."""
    
    def __init__(self, parent: Optional[State] = None):
        """Initialize a new state container.
        
        Args:
            parent: Optional parent state to inherit from.
        """
        self._parent = parent
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._children: List[State] = []
        
        if parent is not None:
            parent._add_child(self)
    
    def _add_child(self, child: State) -> None:
        """Add a child state."""
        with self._lock:
            self._children.append(child)
    
    def _remove_child(self, child: State) -> None:
        """Remove a child state."""
        with self._lock:
            if child in self._children:
                self._children.remove(child)
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get a value from the state.
        
        Args:
            key: The key to look up.
            default: Default value if the key is not found.
            
        Returns:
            The value, or default if not found.
        """
        try:
            return self[key]
        except StateKeyError:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get a value from the state, with support for dot notation."""
        with self._lock:
            # Handle dot notation for nested access
            if '.' in key:
                first, rest = key.split('.', 1)
                if first not in self._data:
                    if self._parent is not None:
                        return self._parent[rest]
                    raise StateKeyError(f"No such key: {first}")
                
                value = self._data[first]
                if isinstance(value, State):
                    return value[rest]
                elif isinstance(value, dict):
                    return _get_nested(value, rest)
                else:
                    raise StateKeyError(f"Cannot access '{rest}' on non-container type {type(value).__name__}")
            
            # Direct access
            if key in self._data:
                entry = self._data[key]
                if isinstance(entry, StateEntry):
                    return entry.value
                return entry
            
            # Check parent if not found
            if self._parent is not None:
                return self._parent[key]
            
            raise StateKeyError(f"No such key: {key}")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value in the state, with support for dot notation."""
        with self._lock:
            # Handle dot notation for nested access
            if '.' in key:
                first, rest = key.split('.', 1)
                
                # Get or create the nested state
                if first not in self._data:
                    self._data[first] = State(parent=self)
                
                nested = self._data[first]
                if isinstance(nested, State):
                    nested[rest] = value
                    return
                elif isinstance(nested, dict):
                    _set_nested(nested, rest, value)
                    return
                else:
                    raise StateKeyError(f"Cannot set '{rest}' on non-container type {type(nested).__name__}")
            
            # Direct access
            if key in self._data and isinstance(self._data[key], StateEntry):
                self._data[key].value = value
            else:
                self._data[key] = value
    
    def __delitem__(self, key: str) -> None:
        """Delete a value from the state."""
        with self._lock:
            if key in self._data:
                if isinstance(self._data[key], StateEntry):
                    self._data[key].clear()
                else:
                    del self._data[key]
            elif self._parent is not None and key in self._parent:
                del self._parent[key]
            else:
                raise StateKeyError(f"No such key: {key}")
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state."""
        with self._lock:
            if key in self._data:
                return True
            if self._parent is not None:
                return key in self._parent
            return False
    
    def keys(self) -> List[str]:
        """Get all keys in the state."""
        with self._lock:
            keys = set(self._data.keys())
            if self._parent is not None:
                keys.update(self._parent.keys())
            return list(keys)
    
    def items(self) -> List[Tuple[str, Any]]:
        """Get all key-value pairs in the state."""
        return [(k, self[k]) for k in self.keys()]
    
    def values(self) -> List[Any]:
        """Get all values in the state."""
        return [self[k] for k in self.keys()]
    
    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple values in the state."""
        with self._lock:
            for key, value in values.items():
                self[key] = value
    
    def clear(self) -> None:
        """Clear all values in the state."""
        with self._lock:
            self._data.clear()
    
    def get_entry(self, key: str) -> Optional[StateEntry[Any]]:
        """Get the StateEntry for a key, if it exists."""
        with self._lock:
            if key in self._data and isinstance(self._data[key], StateEntry):
                return self._data[key]
            return None
    
    def set_entry(
        self,
        key: str,
        value: Optional[T] = None,
        default: Optional[Union[T, Callable[[], T]]] = None,
        validator: Optional[Callable[[T], bool]] = None,
        ttl: Optional[float] = None
    ) -> StateEntry[T]:
        """Set a state entry with validation and TTL.
        
        Args:
            key: The key to set.
            value: Initial value (None means unset).
            default: Default value or factory function.
            validator: Optional validation function.
            ttl: Time-to-live in seconds (None means no expiration).
            
        Returns:
            The created or updated StateEntry.
        """
        with self._lock:
            entry = StateEntry(value=value, default=default, validator=validator, ttl=ttl)
            self._data[key] = entry
            return entry
    
    def create_child(self) -> 'State':
        """Create a new child state that inherits from this one."""
        return State(parent=self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        result = {}
        with self._lock:
            for key in self.keys():
                value = self[key]
                if isinstance(value, State):
                    result[key] = value.to_dict()
                elif hasattr(value, 'dict'):
                    # Handle Pydantic models
                    result[key] = value.dict()
                else:
                    result[key] = value
        return result
    
    def __str__(self) -> str:
        """Get a string representation of the state."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def __enter__(self) -> 'State':
        """Context manager entry."""
        self._token = _current_state.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        _current_state.reset(self._token)
        
        # Clean up any expired entries
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up expired entries."""
        with self._lock:
            keys_to_remove = []
            
            for key, value in self._data.items():
                if isinstance(value, StateEntry) and value.is_expired:
                    keys_to_remove.append(key)
                elif isinstance(value, State):
                    value._cleanup()
            
            for key in keys_to_remove:
                del self._data[key]
            
            # Clean up children
            for child in self._children:
                child._cleanup()


def _get_nested(d: Dict[str, Any], key: str) -> Any:
    """Get a value from a nested dictionary using dot notation."""
    parts = key.split('.')
    current = d
    
    for part in parts:
        if part not in current:
            raise StateKeyError(f"No such key: {key}")
        current = current[part]
    
    return current


def _set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation."""
    parts = key.split('.')
    current = d
    
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
        
        if not isinstance(current, dict):
            raise StateKeyError(f"Cannot set '{key}' - '{part}' is not a dictionary")
    
    current[parts[-1]] = value


def get_current_state() -> State:
    """Get the current state for the current context."""
    state = _current_state.get()
    if state is None:
        raise StateNotInitializedError("No state has been initialized for this context")
    return state


def set_current_state(state: State) -> None:
    """Set the current state for the current context."""
    _current_state.set(state)


@contextlib.contextmanager
def state_context(state: Optional[State] = None) -> Iterator[State]:
    """Context manager for working with a state."""
    if state is None:
        state = State()
    
    token = _current_state.set(state)
    try:
        yield state
    finally:
        _current_state.reset(token)
        state._cleanup()


def state_property(
    default: Optional[Union[T, Callable[[], T]]] = None,
    validator: Optional[Callable[[T], bool]] = None,
    ttl: Optional[float] = None,
    key: Optional[str] = None
) -> Any:
    """Decorator to create a property that reads from/writes to the current state.
    
    Example:
        class MyClass:
            @state_property(default=42, ttl=3600)
            def my_value(self) -> int:
                pass  # The getter is replaced
    """
    def decorator(method: Callable[..., T]) -> property:
        # Use the method name as the key if none provided
        prop_name = key or method.__name__
        
        def getter(self) -> T:
            state = get_current_state()
            full_key = f"{self.__class__.__name__}.{prop_name}"
            
            # Try to get existing entry
            entry = state.get_entry(full_key)
            if entry is not None:
                return entry.value
            
            # Create a new entry with the default value
            if default is not None:
                entry = state.set_entry(
                    key=full_key,
                    default=default,
                    validator=validator,
                    ttl=ttl
                )
                return entry.value
            
            raise StateKeyError(f"No value set for {full_key} and no default provided")
        
        def setter(self, value: T) -> None:
            state = get_current_state()
            full_key = f"{self.__class__.__name__}.{prop_name}"
            
            # Get or create the entry
            entry = state.get_entry(full_key)
            if entry is None:
                entry = state.set_entry(
                    key=full_key,
                    default=default,
                    validator=validator,
                    ttl=ttl
                )
            
            entry.value = value
        
        def deleter(self) -> None:
            state = get_current_state()
            full_key = f"{self.__class__.__name__}.{prop_name}"
            
            if full_key in state:
                del state[full_key]
        
        return property(getter, setter, deleter, method.__doc__)
    
    return decorator


# Default global state
global_state = State()

# Set as the default current state
_current_state.set(global_state)

# Helper functions for working with the current state
def get(key: str, default: Optional[T] = None) -> Optional[T]:
    """Get a value from the current state."""
    return get_current_state().get(key, default)

def set(key: str, value: Any) -> None:
    """Set a value in the current state."""
    get_current_state()[key] = value

def delete(key: str) -> None:
    """Delete a value from the current state."""
    if key in get_current_state():
        del get_current_state()[key]
