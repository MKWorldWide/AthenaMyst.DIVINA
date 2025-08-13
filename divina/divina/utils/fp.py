""
Functional programming utilities and type-safe decorators.
"""
import functools
import inspect
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from loguru import logger

T = TypeVar('T')
R = TypeVar('R')


def pure(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to enforce pure functions (no side effects).
    
    This is a runtime check to help catch accidental side effects in functions
    that should be pure. It's not foolproof but can help during development.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Check for mutable default arguments
        if func.__defaults__:
            for i, default in enumerate(func.__defaults__):
                if isinstance(default, (list, dict, set)):
                    param_name = func.__code__.co_varnames[i]
                    logger.warning(
                        f"Function {func.__name__} has mutable default argument "
                        f"'{param_name}'. This can lead to unexpected behavior."
                    )
        
        # Get the initial state of the module's globals
        module_globals = func.__globals__.copy()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Check for modifications to globals
        current_globals = func.__globals__
        modified_globals = {
            k: (module_globals[k], current_globals[k])
            for k in module_globals
            if (k in current_globals and 
                k not in {'__name__', '__file__', '__cached__', '__builtins__', '__package__'} and
                module_globals[k] != current_globals[k])
        }
        
        if modified_globals:
            logger.warning(
                f"Function {func.__name__} modified global state: {modified_globals}"
            )
        
        return result
    
    return wrapper


def memoize(maxsize: int = 128) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Memoization decorator with LRU cache.
    
    This is a pure-Python implementation that works with unhashable arguments.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        cache: Dict[str, R] = {}
        cache_keys: List[str] = []
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Create a cache key from the function arguments
            key_parts = [
                str(arg) if isinstance(arg, (int, float, str, bool, type(None))) 
                else str(id(arg))
                for arg in args
            ]
            key_parts.extend(
                f"{k}={v}" if isinstance(v, (int, float, str, bool, type(None))) 
                else f"{k}={id(v)}"
                for k, v in sorted(kwargs.items())
            )
            key = ":".join(key_parts)
            
            # Check cache
            if key in cache:
                return cache[key]
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Update cache
            cache[key] = result
            cache_keys.append(key)
            
            # Enforce maxsize
            if len(cache) > maxsize:
                oldest_key = cache_keys.pop(0)
                if oldest_key in cache:
                    del cache[oldest_key]
            
            return result
        
        return wrapper
    
    return decorator


def ensure_utc(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to ensure datetime arguments are timezone-aware and in UTC."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Get the function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        
        # Check datetime arguments
        for param_name, param in sig.parameters.items():
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                
                # Handle datetime objects
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        logger.warning(
                            f"Function {func.__name__} received naive datetime for "
                            f"parameter '{param_name}'. Assuming UTC."
                        )
                        bound_args.arguments[param_name] = value.replace(tzinfo=timezone.utc)
                    elif value.tzinfo != timezone.utc:
                        logger.warning(
                            f"Function {func.__name__} received timezone-aware datetime "
                            f"for parameter '{param_name}' that's not in UTC. Converting to UTC."
                        )
                        bound_args.arguments[param_name] = value.astimezone(timezone.utc)
                
                # Handle lists/tuples of datetimes
                elif isinstance(value, (list, tuple)) and value and isinstance(value[0], datetime):
                    for i, dt in enumerate(value):
                        if isinstance(dt, datetime):
                            if dt.tzinfo is None:
                                logger.warning(
                                    f"Function {func.__name__} received naive datetime in list "
                                    f"for parameter '{param_name}'. Assuming UTC."
                                )
                                value = list(value)  # Convert to list if it's a tuple
                                value[i] = dt.replace(tzinfo=timezone.utc)
                            elif dt.tzinfo != timezone.utc:
                                logger.warning(
                                    f"Function {func.__name__} received timezone-aware datetime "
                                    f"in list for parameter '{param_name}' that's not in UTC. "
                                    "Converting to UTC."
                                )
                                value = list(value)  # Convert to list if it's a tuple
                                value[i] = dt.astimezone(timezone.utc)
                    
                    bound_args.arguments[param_name] = value
        
        # Call the function with processed arguments
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper


def validate_types(strict: bool = False) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator to validate function argument and return types using type hints.
    
    Args:
        strict: If True, raises TypeError on type mismatch. If False, logs a warning.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        # Get type hints
        type_hints = {}
        if hasattr(func, '__annotations__'):
            type_hints = func.__annotations__.copy()
        
        # Get return type
        return_type = type_hints.pop('return', None)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Check argument types
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            for param_name, param in sig.parameters.items():
                if param_name in bound_args.arguments and param_name in type_hints:
                    value = bound_args.arguments[param_name]
                    expected_type = type_hints[param_name]
                    
                    # Handle Optional types
                    if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                        # Check if None is one of the options (Optional)
                        if type(None) in expected_type.__args__:
                            # Get the non-None type
                            non_none_types = [t for t in expected_type.__args__ if t is not type(None)]
                            if not non_none_types:
                                continue
                            expected_type = Union[tuple(non_none_types)]
                    
                    # Handle List, Dict, etc.
                    if hasattr(expected_type, '__origin__'):
                        origin = expected_type.__origin__
                        
                        # Handle List[T], Tuple[T], etc.
                        if origin in (list, List) and isinstance(value, list):
                            if expected_type.__args__:
                                item_type = expected_type.__args__[0]
                                for i, item in enumerate(value):
                                    if not isinstance(item, item_type):
                                        msg = (
                                            f"Argument '{param_name}[{i}]' has incorrect type. "
                                            f"Expected {item_type}, got {type(item).__name__}"
                                        )
                                        if strict:
                                            raise TypeError(msg)
                                        logger.warning(msg)
                        
                        # Handle Dict[K, V]
                        elif origin in (dict, Dict) and isinstance(value, dict):
                            if len(expected_type.__args__) >= 2:
                                key_type, value_type = expected_type.__args__[:2]
                                for k, v in value.items():
                                    if not isinstance(k, key_type) or not isinstance(v, value_type):
                                        msg = (
                                            f"Argument '{param_name}' has incorrect dict item types. "
                                            f"Expected Dict[{key_type.__name__}, {value_type.__name__}], "
                                            f"got Dict[{type(k).__name__}, {type(v).__name__}]"
                                        )
                                        if strict:
                                            raise TypeError(msg)
                                        logger.warning(msg)
                        
                        # Handle other generic types (fallback)
                        elif not isinstance(value, origin):
                            msg = (
                                f"Argument '{param_name}' has incorrect type. "
                                f"Expected {expected_type}, got {type(value).__name__}"
                            )
                            if strict:
                                raise TypeError(msg)
                            logger.warning(msg)
                    
                    # Handle regular types
                    elif not isinstance(value, expected_type):
                        msg = (
                            f"Argument '{param_name}' has incorrect type. "
                            f"Expected {expected_type.__name__}, got {type(value).__name__}"
                        )
                        if strict:
                            raise TypeError(msg)
                        logger.warning(msg)
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Check return type
            if return_type is not None and not isinstance(result, return_type):
                msg = (
                    f"Return value has incorrect type. "
                    f"Expected {return_type.__name__}, got {type(result).__name__}"
                )
                if strict:
                    raise TypeError(msg)
                logger.warning(msg)
            
            return result
        
        return wrapper
    
    return decorator


def no_global_state(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to prevent functions from modifying global state.
    
    This is a runtime check that raises an exception if the function tries to
    modify global variables.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Get the function's globals
        global_vars = func.__globals__.copy()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Check for modifications to globals
        current_globals = func.__globals__
        modified_globals = {
            k: (global_vars[k], current_globals[k])
            for k in global_vars
            if (k in current_globals and 
                k not in {'__name__', '__file__', '__cached__', '__builtins__', '__package__'} and
                global_vars[k] != current_globals[k])
        }
        
        if modified_globals:
            raise RuntimeError(
                f"Function {func.__name__} modified global state: {modified_globals}"
            )
        
        return result
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Example of using the decorators
    
    @pure
    @ensure_utc
    @validate_types()
    def process_timestamps(timestamps: List[datetime]) -> List[datetime]:
        """Example function that processes timestamps."""
        # This function is pure and type-safe
        return [ts.replace(second=0) for ts in timestamps]
    
    # This will work
    now = datetime.now(timezone.utc)
    result = process_timestamps([now])
    print(f"Processed timestamp: {result[0]}")
    
    # This will log a warning about timezone
    naive_time = datetime(2023, 1, 1, 12, 0, 0)
    result = process_timestamps([naive_time])
    print(f"Processed naive timestamp: {result[0]}")
    
    # This will raise a type error in strict mode
    try:
        @validate_types(strict=True)
        def add(a: int, b: int) -> int:
            return a + b
        
        add(1, "2")  # Will raise TypeError
    except TypeError as e:
        print(f"Caught type error: {e}")
