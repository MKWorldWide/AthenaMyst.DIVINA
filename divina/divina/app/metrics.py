""
Metrics collection and export for monitoring and observability.

Uses Prometheus client to expose metrics via HTTP endpoint.
"""
from typing import Dict, Any, Optional, Tuple
from time import time
from fastapi import Request, Response, APIRouter
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR
)

# Remove default collectors
for collector in list(REGISTRY._collector_to_names.keys()):
    if collector in [PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR]:
        REGISTRY.unregister(collector)

# Create metrics
REQUEST_COUNT = Counter(
    'divina_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'divina_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

SIGNAL_COUNT = Counter(
    'divina_signals_total',
    'Total number of trading signals generated',
    ['pair', 'timeframe', 'signal_type']
)

ALERT_COUNT = Counter(
    'divina_alerts_total',
    'Total number of alerts sent',
    ['alert_type', 'status']
)

CANDLE_COUNT = Counter(
    'divina_candles_processed_total',
    'Total number of candles processed',
    ['pair', 'timeframe']
)

# System metrics
MEMORY_USAGE = Gauge(
    'divina_memory_usage_bytes',
    'Current memory usage in bytes'
)

CPU_USAGE = Gauge(
    'divina_cpu_usage_percent',
    'Current CPU usage percentage'
)

# Create router
metrics_router = APIRouter()


def record_metrics(request: Request, response: Response, process_time: float) -> None:
    """Record metrics for a request."""
    # Skip metrics for certain paths
    if request.url.path in ['/metrics', '/healthz']:
        return
    
    # Get endpoint name (use route path if available)
    endpoint = request.url.path
    if request.scope.get('route'):
        endpoint = request.scope['route'].path  # type: ignore
    
    # Record request metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(process_time)


def record_signal(pair: str, timeframe: str, signal_type: str) -> None:
    """Record a trading signal."""
    SIGNAL_COUNT.labels(
        pair=pair,
        timeframe=timeframe,
        signal_type=signal_type
    ).inc()


def record_alert(alert_type: str, status: str = 'success') -> None:
    """Record an alert."""
    ALERT_COUNT.labels(
        alert_type=alert_type,
        status=status
    ).inc()


def record_candles(pair: str, timeframe: str, count: int = 1) -> None:
    """Record processed candles."""
    CANDLE_COUNT.labels(
        pair=pair,
        timeframe=timeframe
    ).inc(count)


@metrics_router.get("/", include_in_schema=False)
async def get_metrics() -> Response:
    """Expose Prometheus metrics."""
    # Update system metrics
    try:
        import psutil
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent())
    except ImportError:
        pass
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Add custom collectors for application-specific metrics
class SignalMetrics:
    """Custom collector for signal metrics."""
    
    def collect(self) -> None:
        """Collect signal metrics."""
        yield from SIGNAL_COUNT.collect()
        yield from ALERT_COUNT.collect()
        yield from CANDLE_COUNT.collect()


# Register custom collectors
REGISTRY.register(SignalMetrics())
