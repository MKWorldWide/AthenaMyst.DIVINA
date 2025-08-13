""
Monitoring and observability for AthenaMyst:Divina.

This module provides health checks, metrics, and monitoring functionality.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.metrics import MetricWrapperBase
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, Response as StarletteResponse

from ..alerting import AlertLevel, send_alert
from ..state import State, state_context
from ..utils.time_utils import now_utc


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)


class HealthCheck:
    ""
    A health check that can be registered with the HealthMonitor.
    
    Example:
        async def check_db():
            try:
                await db.ping()
                return HealthCheckResult("database", HealthStatus.HEALTHY)
            except Exception as e:
                return HealthCheckResult("database", HealthStatus.UNHEALTHY, str(e))
                
        health_check = HealthCheck("database", check_db, timeout=5.0, interval=60.0)
    """
    
    def __init__(
        self,
        name: str,
        check_func: Callable[..., HealthCheckResult],
        timeout: float = 10.0,
        interval: float = 30.0,
        critical: bool = True,
        **check_kwargs,
    ):
        """Initialize a health check.
        
        Args:
            name: Unique name for this health check.
            check_func: Async function that performs the health check.
            timeout: Maximum time in seconds to wait for the check to complete.
            interval: How often to run this check in seconds.
            critical: If True, a failing check will make the overall status UNHEALTHY.
                     If False, it will only make the status DEGRADED.
            **check_kwargs: Additional keyword arguments to pass to the check function.
        """
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.interval = interval
        self.critical = critical
        self.check_kwargs = check_kwargs
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[HealthCheckResult] = None
        self._task: Optional[asyncio.Task] = None
    
    async def run(self) -> HealthCheckResult:
        """Run the health check."""
        self._last_run = now_utc()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                self.check_func(**self.check_kwargs),
                timeout=self.timeout,
            )
            
            # Ensure the result has the correct name
            result.name = self.name
            
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout} seconds",
            )
        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )
        
        self._last_result = result
        return result
    
    @property
    def last_run(self) -> Optional[datetime]:
        """When this health check was last run."""
        return self._last_run
    
    @property
    def last_result(self) -> Optional[HealthCheckResult]:
        """Result of the last run of this health check."""
        return self._last_result
    
    @property
    def is_stale(self) -> bool:
        """Whether this health check is stale (not run recently)."""
        if self._last_run is None:
            return True
        return (now_utc() - self._last_run).total_seconds() > self.interval * 1.5
    
    def start(self) -> None:
        """Start running this health check periodically."""
        if self._task is not None and not self._task.done():
            return
        
        async def run_periodically():
            while True:
                try:
                    await self.run()
                except Exception as e:
                    logger.error(f"Error in health check {self.name}: {e}")
                
                await asyncio.sleep(self.interval)
        
        self._task = asyncio.create_task(run_periodically())
    
    def stop(self) -> None:
        """Stop running this health check."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None


class HealthMonitor:
    """Manages health checks and provides health status."""
    
    def __init__(self, name: str = "athenamyst"):
        """Initialize the health monitor."""
        self.name = name
        self.checks: Dict[str, HealthCheck] = {}
        self._started = False
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        if check.name in self.checks:
            raise ValueError(f"Health check with name '{check.name}' already exists")
        
        self.checks[check.name] = check
        
        # Start the check if the monitor is already running
        if self._started:
            check.start()
    
    def remove_check(self, name: str) -> None:
        """Remove a health check by name."""
        if name in self.checks:
            self.checks[name].stop()
            del self.checks[name]
    
    async def run_checks(self, names: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks and return the results.
        
        Args:
            names: Optional list of check names to run. If None, runs all checks.
            
        Returns:
            Dictionary mapping check names to their results.
        """
        if names is None:
            checks_to_run = list(self.checks.values())
        else:
            checks_to_run = [self.checks[name] for name in names if name in self.checks]
        
        # Run checks in parallel
        results = await asyncio.gather(
            *(check.run() for check in checks_to_run),
            return_exceptions=True,
        )
        
        # Process results
        result_dict = {}
        for check, result in zip(checks_to_run, results):
            if isinstance(result, Exception):
                result_dict[check.name] = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {result}",
                    details={"error": str(result)},
                )
            else:
                result_dict[check.name] = result
        
        return result_dict
    
    async def get_status(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get the current health status.
        
        Args:
            names: Optional list of check names to include. If None, includes all checks.
            
        Returns:
            Dictionary with overall status and check results.
        """
        if names is None:
            checks = list(self.checks.values())
            results = {check.name: check.last_result for check in checks}
        else:
            results = {}
            for name in names:
                if name in self.checks:
                    results[name] = self.checks[name].last_result
        
        # Determine overall status
        status = HealthStatus.HEALTHY
        messages = []
        
        for check_name, result in results.items():
            if result is None:
                continue
                
            check = self.checks[check_name]
            
            if check.is_stale:
                messages.append(f"{check_name}: check is stale (last run: {check.last_run})")
                status = max(status, HealthStatus.DEGRADED)
            
            if result.status == HealthStatus.UNHEALTHY:
                if check.critical:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = max(status, HealthStatus.DEGRADED)
                messages.append(f"{check_name}: {result.message}")
            elif result.status == HealthStatus.DEGRADED:
                status = max(status, HealthStatus.DEGRADED)
                messages.append(f"{check_name}: {result.message}")
        
        return {
            "status": status,
            "name": self.name,
            "timestamp": now_utc().isoformat(),
            "checks": {
                name: result.dict() if result else None
                for name, result in results.items()
            },
            "message": "; ".join(messages) if messages else "All systems operational",
        }
    
    def start(self) -> None:
        """Start all health checks."""
        if self._started:
            return
        
        for check in self.checks.values():
            check.start()
        
        self._started = True
    
    def stop(self) -> None:
        """Stop all health checks."""
        for check in self.checks.values():
            check.stop()
        
        self._started = False


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""
    
    def __init__(
        self,
        app,
        app_name: str = "athenamyst",
        include_path: bool = True,
        include_method: bool = True,
        include_status: bool = True,
        include_duration: bool = True,
        include_client: bool = False,
        custom_labels: Optional[Dict[str, str]] = None,
        skip_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the middleware."""
        super().__init__(app, **kwargs)
        
        self.app_name = app_name
        self.include_path = include_path
        self.include_method = include_method
        self.include_status = include_status
        self.include_duration = include_duration
        self.include_client = include_client
        self.custom_labels = custom_labels or {}
        self.skip_paths = set(skip_paths or [])
        
        # Initialize metrics
        self.request_counter = Counter(
            f"{self.app_name}_requests_total",
            "Total number of requests",
            ["method", "path", "status"],
        )
        
        self.request_duration = Histogram(
            f"{self.app_name}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "path", "status"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, float("inf")),
        )
        
        self.request_size = Histogram(
            f"{self.app_name}_request_size_bytes",
            "Request size in bytes",
            ["method", "path"],
            buckets=(100, 1000, 10000, 100000, 1000000, float("inf")),
        )
        
        self.response_size = Histogram(
            f"{self.app_name}_response_size_bytes",
            "Response size in bytes",
            ["method", "path", "status"],
            buckets=(100, 1000, 10000, 100000, 1000000, float("inf")),
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            f"{self.app_name}_cpu_usage_percent",
            "CPU usage percentage",
            ["cpu"],
        )
        
        self.memory_usage = Gauge(
            f"{self.app_name}_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],
        )
        
        self.disk_usage = Gauge(
            f"{self.app_name}_disk_usage_bytes",
            "Disk usage in bytes",
            ["device", "mountpoint"],
        )
        
        self.process_metrics = Gauge(
            f"{self.app_name}_process",
            "Process metrics",
            ["metric"],
        )
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        """Process the request and update metrics."""
        # Skip metrics for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Track request start time
        start_time = time.monotonic()
        
        # Get request size
        request_size = 0
        if hasattr(request, "_receive"):
            # For ASGI requests, we need to read the body to get the size
            body = b""
            async for chunk in request.stream():
                body += chunk
            request_size = len(body)
            # Create a new request with the consumed body
            request = Request(scope=request.scope, receive=request.receive, _body=body)
        
        # Process the request
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.monotonic() - start_time
        
        # Get response size
        response_size = 0
        if hasattr(response, "body"):
            response_size = len(response.body)
        elif hasattr(response, "body_iterator"):
            # For streaming responses, we can't easily get the size
            response_size = -1
        
        # Update metrics
        labels = {}
        if self.include_method:
            labels["method"] = request.method
        if self.include_path:
            labels["path"] = request.url.path
        if self.include_status:
            labels["status"] = str(response.status_code)
        
        # Update request counter
        self.request_counter.labels(**labels).inc()
        
        # Update duration histogram
        if self.include_duration:
            self.request_duration.labels(**labels).observe(duration)
        
        # Update size metrics
        if request_size > 0:
            self.request_size.labels(
                method=request.method,
                path=request.url.path,
            ).observe(request_size)
        
        if response_size >= 0:
            self.response_size.labels(
                method=request.method,
                path=request.url.path,
                status=str(response.status_code),
            ).observe(response_size)
        
        # Update system metrics
        self._update_system_metrics()
        
        return response
    
    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        # CPU usage
        for cpu, percent in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
            self.cpu_usage.labels(cpu=f"cpu{cpu}").set(percent)
        
        # Memory usage
        mem = psutil.virtual_memory()
        self.memory_usage.labels(type="used").set(mem.used)
        self.memory_usage.labels(type="available").set(mem.available)
        self.memory_usage.labels(type="total").set(mem.total)
        
        # Disk usage
        for part in psutil.disk_partitions(all=False):
            if not part.mountpoint:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
                self.disk_usage.labels(
                    device=part.device,
                    mountpoint=part.mountpoint,
                ).set(usage.used)
            except Exception as e:
                logger.warning(f"Failed to get disk usage for {part.mountpoint}: {e}")
        
        # Process metrics
        process = psutil.Process()
        self.process_metrics.labels("cpu_percent").set(process.cpu_percent())
        self.process_metrics.labels("memory_rss").set(process.memory_info().rss)
        self.process_metrics.labels("memory_vms").set(process.memory_info().vms)
        self.process_metrics.labels("threads").set(process.num_threads())
        self.process_metrics.labels("fds").set(process.num_fds() if hasattr(process, 'num_fds') else -1)


def create_monitoring_router(
    health_monitor: Optional[HealthMonitor] = None,
    metrics_middleware: Optional[MetricsMiddleware] = None,
    prefix: str = "",
) -> APIRouter:
    """Create a FastAPI router with health and metrics endpoints.
    
    Args:
        health_monitor: Optional HealthMonitor instance.
        metrics_middleware: Optional MetricsMiddleware instance.
        prefix: URL prefix for all routes.
        
    Returns:
        FastAPI router with monitoring endpoints.
    """
    router = APIRouter(prefix=prefix)
    
    @router.get("/healthz", tags=["monitoring"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        if health_monitor is None:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "Health monitor not configured",
                "timestamp": now_utc().isoformat(),
            }
        
        return await health_monitor.get_status()
    
    @router.get("/metrics", tags=["monitoring"])
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        # Update system metrics if we have a metrics middleware
        if metrics_middleware is not None:
            metrics_middleware._update_system_metrics()
        
        # Generate the metrics response
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    @router.get("/version", tags=["monitoring"])
    async def version() -> Dict[str, str]:
        """Get application version information."""
        import pkg_resources
        
        try:
            version = pkg_resources.get_distribution("athenamyst").version
        except pkg_resources.DistributionNotFound:
            version = "unknown"
        
        return {
            "name": "AthenaMyst:Divina",
            "version": version,
            "python": "{}.{}.{}".format(*sys.version_info[:3]),
            "platform": sys.platform,
        }
    
    @router.get("/status", tags=["monitoring"])
    async def status() -> Dict[str, Any]:
        """Get detailed status information."""
        import platform
        
        # Get process info
        process = psutil.Process()
        mem_info = process.memory_full_info()
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "ok",
            "timestamp": now_utc().isoformat(),
            "system": {
                "python": {
                    "version": "{}.{}.{}".format(*sys.version_info[:3]),
                    "implementation": platform.python_implementation(),
                    "compiler": platform.python_compiler(),
                },
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg(),
                },
                "memory": {
                    "total": mem.total,
                    "available": mem.available,
                    "percent": mem.percent,
                    "used": mem.used,
                    "free": mem.free,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
            },
            "process": {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                "cpu": {
                    "percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                },
                "memory": {
                    "rss": mem_info.rss,
                    "vms": mem_info.vms,
                    "shared": getattr(mem_info, 'shared', 0),
                    "text": getattr(mem_info, 'text', 0),
                    "lib": getattr(mem_info, 'lib', 0),
                    "data": getattr(mem_info, 'data', 0),
                    "dirty": getattr(mem_info, 'dirty', 0),
                },
                "connections": len(process.connections()),
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
            },
        }
    
    return router


# Default global instances
health_monitor = HealthMonitor()
metrics_middleware = None  # Must be initialized with FastAPI app


def init_monitoring(app, **kwargs) -> Tuple[HealthMonitor, MetricsMiddleware]:
    """Initialize monitoring for a FastAPI application.
    
    Args:
        app: FastAPI application instance.
        **kwargs: Additional arguments to pass to MetricsMiddleware.
        
    Returns:
        Tuple of (health_monitor, metrics_middleware).
    """
    global health_monitor, metrics_middleware
    
    # Initialize metrics middleware
    metrics_middleware = MetricsMiddleware(app, **kwargs)
    
    # Add middleware to the app
    app.add_middleware(MetricsMiddleware, **kwargs)
    
    # Add monitoring routes
    monitoring_router = create_monitoring_router(
        health_monitor=health_monitor,
        metrics_middleware=metrics_middleware,
        prefix="/api/monitoring",
    )
    app.include_router(monitoring_router)
    
    # Start health checks
    health_monitor.start()
    
    return health_monitor, metrics_middleware


def add_health_check(
    name: str,
    check_func: Callable[..., HealthCheckResult],
    **kwargs,
) -> HealthCheck:
    """Add a health check to the global health monitor."""
    check = HealthCheck(name, check_func, **kwargs)
    health_monitor.add_check(check)
    return check


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return health_monitor


def get_metrics_middleware() -> Optional[MetricsMiddleware]:
    """Get the global metrics middleware instance, if initialized."""
    return metrics_middleware
