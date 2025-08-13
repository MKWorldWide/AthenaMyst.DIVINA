""
OpenTelemetry integration for AthenaMyst:Divina.

This module provides OpenTelemetry instrumentation for tracing and metrics,
with the ability to toggle it on/off via configuration.
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple, Type, Union, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, export
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware as ASGIMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from pydantic import BaseModel, Field, validator

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class OTLPConfig(BaseModel):
    """OTLP (OpenTelemetry Protocol) configuration."""
    
    endpoint: str = Field(
        "http://localhost:4317",
        description="OTLP gRPC endpoint"
    )
    insecure: bool = Field(
        True,
        description="Whether to use insecure connection"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to include in requests"
    )
    timeout: int = Field(
        10,
        description="Timeout in seconds for OTLP exports"
    )


class JaegerConfig(BaseModel):
    """Jaeger configuration."""
    
    agent_host_name: str = Field(
        "localhost",
        description="Jaeger agent hostname"
    )
    agent_port: int = Field(
        6831,
        description="Jaeger agent port"
    )
    udp_split_oversized_batches: bool = Field(
        True,
        description="Whether to split oversized batches"
    )


class ZipkinConfig(BaseModel):
    """Zipkin configuration."""
    
    endpoint: str = Field(
        "http://localhost:9411/api/v2/spans",
        description="Zipkin collector endpoint"
    )
    timeout: int = Field(
        10,
        description="Timeout in seconds for Zipkin exports"
    )


class ConsoleConfig(BaseModel):
    """Console exporter configuration."""
    
    pretty: bool = Field(
        True,
        description="Whether to pretty-print output"
    )


class TraceConfig(BaseModel):
    """Tracing configuration."""
    
    enabled: bool = Field(
        False,
        description="Enable tracing"
    )
    sampler: str = Field(
        "parentbased_always_on",
        description="Sampler to use (always_on, always_off, traceidratio, parentbased_always_on, etc.)"
    )
    sampler_arg: float = Field(
        1.0,
        description="Sampler argument (e.g., ratio for traceidratio)"
    )
    max_export_batch_size: int = Field(
        512,
        description="Maximum batch size for span exports"
    )
    schedule_delay_millis: int = Field(
        5000,
        description="Delay in milliseconds between export cycles"
    )
    export_timeout_millis: int = Field(
        30000,
        description="Maximum time in milliseconds to wait for an export to complete"
    )
    max_queue_size: int = Field(
        2048,
        description="Maximum queue size for spans before dropping"
    )
    otlp: OTLPConfig = Field(
        default_factory=OTLPConfig,
        description="OTLP exporter configuration"
    )
    jaeger: JaegerConfig = Field(
        default_factory=JaegerConfig,
        description="Jaeger exporter configuration"
    )
    zipkin: ZipkinConfig = Field(
        default_factory=ZipkinConfig,
        description="Zipkin exporter configuration"
    )
    console: ConsoleConfig = Field(
        default_factory=ConsoleConfig,
        description="Console exporter configuration"
    )


class MetricConfig(BaseModel):
    """Metrics configuration."""
    
    enabled: bool = Field(
        False,
        description="Enable metrics"
    )
    export_interval_millis: int = Field(
        60000,
        description="Interval in milliseconds between metric exports"
    )
    export_timeout_millis: int = Field(
        30000,
        description="Timeout in milliseconds for metric exports"
    )
    otlp: OTLPConfig = Field(
        default_factory=OTLPConfig,
        description="OTLP exporter configuration"
    )
    console: ConsoleConfig = Field(
        default_factory=ConsoleConfig,
        description="Console exporter configuration"
    )


class LogCorrelationConfig(BaseModel):
    """Log correlation configuration."""
    
    enabled: bool = Field(
        True,
        description="Enable log correlation with traces"
    )
    log_level: str = Field(
        "INFO",
        description="Log level for OpenTelemetry logs"
    )


class InstrumentationConfig(BaseModel):
    """Instrumentation configuration."""
    
    aiohttp: bool = Field(
        True,
        description="Enable aiohttp instrumentation"
    )
    fastapi: bool = Field(
        True,
        description="Enable FastAPI instrumentation"
    )
    psycopg2: bool = Field(
        True,
        description="Enable psycopg2 instrumentation"
    )
    requests: bool = Field(
        True,
        description="Enable requests instrumentation"
    )
    sqlalchemy: bool = Field(
        True,
        description="Enable SQLAlchemy instrumentation"
    )


class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry configuration."""
    
    service_name: str = Field(
        "athenamyst-divina",
        description="Service name for telemetry"
    )
    service_version: str = Field(
        "0.1.0",
        description="Service version for telemetry"
    )
    environment: str = Field(
        "development",
        description="Deployment environment (e.g., development, staging, production)"
    )
    trace: TraceConfig = Field(
        default_factory=TraceConfig,
        description="Tracing configuration"
    )
    metric: MetricConfig = Field(
        default_factory=MetricConfig,
        description="Metrics configuration"
    )
    log_correlation: LogCorrelationConfig = Field(
        default_factory=LogCorrelationConfig,
        description="Log correlation configuration"
    )
    instrumentation: InstrumentationConfig = Field(
        default_factory=InstrumentationConfig,
        description="Instrumentation configuration"
    )
    resource_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional resource attributes"
    )


class OpenTelemetryManager:
    """Manages OpenTelemetry instrumentation and exporters."""
    
    def __init__(self, config: Optional[OpenTelemetryConfig] = None):
        """Initialize the OpenTelemetry manager."""
        self.config = config or self._load_config()
        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._span_exporters: Dict[str, SpanExporter] = {}
        self._metric_exporters: Dict[str, MetricExporter] = {}
        self._initialized = False
    
    @classmethod
    def _load_config(cls) -> OpenTelemetryConfig:
        """Load OpenTelemetry configuration from environment variables."""
        # Load from environment variables with OTEL_ prefix
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith("OTEL_"):
                # Convert OTEL_SOME_KEY to some_key
                config_key = key[5:].lower()
                env_config[config_key] = value
        
        # Create config with defaults, overridden by environment variables
        return OpenTelemetryConfig(**env_config)
    
    def is_enabled(self) -> bool:
        """Check if OpenTelemetry is enabled."""
        return self.config.trace.enabled or self.config.metric.enabled
    
    def _setup_resource(self) -> Resource:
        """Create a resource with service metadata."""
        attributes = {
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "environment": self.config.environment,
            **self.config.resource_attributes,
        }
        return Resource(attributes=attributes)
    
    def _create_span_exporters(self) -> Dict[str, SpanExporter]:
        """Create configured span exporters."""
        exporters = {}
        
        # OTLP exporter
        if "otlp" in os.environ.get("OTEL_TRACES_EXPORTER", "").lower():
            exporters["otlp"] = OTLPSpanExporter(
                endpoint=self.config.trace.otlp.endpoint,
                insecure=self.config.trace.otlp.insecure,
                headers=tuple(self.config.trace.otlp.headers.items()),
                timeout=self.config.trace.otlp.timeout,
            )
        
        # Jaeger exporter
        if "jaeger" in os.environ.get("OTEL_TRACES_EXPORTER", "").lower():
            exporters["jaeger"] = JaegerExporter(
                agent_host_name=self.config.trace.jaeger.agent_host_name,
                agent_port=self.config.trace.jaeger.agent_port,
                udp_split_oversized_batches=self.config.trace.jaeger.udp_split_oversized_batches,
            )
        
        # Zipkin exporter
        if "zipkin" in os.environ.get("OTEL_TRACES_EXPORTER", "").lower():
            exporters["zipkin"] = ZipkinExporter(
                endpoint=self.config.trace.zipkin.endpoint,
                timeout=self.config.trace.zipkin.timeout,
            )
        
        # Console exporter (for development)
        if "console" in os.environ.get("OTEL_TRACES_EXPORTER", "").lower():
            exporters["console"] = ConsoleSpanExporter()
        
        return exporters
    
    def _create_metric_exporters(self) -> Dict[str, MetricExporter]:
        """Create configured metric exporters."""
        exporters = {}
        
        # OTLP exporter
        if "otlp" in os.environ.get("OTEL_METRICS_EXPORTER", "").lower():
            exporters["otlp"] = OTLPMetricExporter(
                endpoint=self.config.metric.otlp.endpoint,
                insecure=self.config.metric.otlp.insecure,
                headers=tuple(self.config.metric.otlp.headers.items()),
                timeout=self.config.metric.otlp.timeout,
            )
        
        # Console exporter (for development)
        if "console" in os.environ.get("OTEL_METRICS_EXPORTER", "").lower():
            exporters["console"] = ConsoleMetricExporter()
        
        return exporters
    
    def _setup_tracing(self) -> None:
        """Set up tracing with configured exporters."""
        if not self.config.trace.enabled:
            logger.debug("Tracing is disabled")
            return
        
        # Create resource
        resource = self._setup_resource()
        
        # Create tracer provider
        self._tracer_provider = TracerProvider(
            resource=resource,
        )
        
        # Configure sampler
        sampler = self._create_sampler()
        if sampler:
            self._tracer_provider = TracerProvider(
                resource=resource,
                sampler=sampler,
            )
        
        # Create and register span exporters
        self._span_exporters = self._create_span_exporters()
        
        for name, exporter in self._span_exporters.items():
            span_processor = BatchSpanProcessor(
                exporter,
                schedule_delay_millis=self.config.trace.schedule_delay_millis,
                max_export_batch_size=self.config.trace.max_export_batch_size,
                export_timeout_millis=self.config.trace.export_timeout_millis,
                max_queue_size=self.config.trace.max_queue_size,
            )
            self._tracer_provider.add_span_processor(span_processor)
            logger.debug(f"Added {name} span exporter")
        
        # Set as global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        
        logger.info("Tracing configured")
    
    def _create_sampler(self):
        """Create a sampler based on configuration."""
        sampler_name = self.config.trace.sampler.lower()
        sampler_arg = self.config.trace.sampler_arg
        
        if sampler_name == "always_on":
            return trace.ALWAYS_ON
        elif sampler_name == "always_off":
            return trace.ALWAYS_OFF
        elif sampler_name == "traceidratio":
            return trace.TraceIdRatioBased(ratio=sampler_arg)
        elif sampler_name == "parentbased_always_on":
            return trace.ParentBased(root=trace.ALWAYS_ON)
        elif sampler_name == "parentbased_always_off":
            return trace.ParentBased(root=trace.ALWAYS_OFF)
        elif sampler_name == "parentbased_traceidratio":
            return trace.ParentBased(root=trace.TraceIdRatioBased(ratio=sampler_arg))
        else:
            logger.warning(f"Unknown sampler: {sampler_name}, using default")
            return trace.ParentBased(root=trace.ALWAYS_ON)
    
    def _setup_metrics(self) -> None:
        """Set up metrics with configured exporters."""
        if not self.config.metric.enabled:
            logger.debug("Metrics are disabled")
            return
        
        # Create resource
        resource = self._setup_resource()
        
        # Create metric readers
        readers = []
        self._metric_exporters = self._create_metric_exporters()
        
        for name, exporter in self._metric_exporters.items():
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metric.export_interval_millis,
                export_timeout_millis=self.config.metric.export_timeout_millis,
            )
            readers.append(reader)
            logger.debug(f"Added {name} metric exporter")
        
        # Create and set meter provider
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers,
        )
        set_meter_provider(self._meter_provider)
        
        logger.info("Metrics configured")
    
    def _setup_log_correlation(self) -> None:
        """Set up log correlation with traces."""
        if not self.config.log_correlation.enabled:
            return
        
        # Configure OpenTelemetry logging instrumentation
        LoggingInstrumentor().instrument(
            set_logging_format=True,
            log_level=self.config.log_correlation.log_level,
        )
        
        logger.info("Log correlation configured")
    
    def _setup_instrumentation(self) -> None:
        """Set up instrumentation for supported libraries."""
        if self.config.instrumentation.aiohttp:
            try:
                AioHttpClientInstrumentor().instrument()
                logger.debug("aiohttp instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument aiohttp: {e}")
        
        if self.config.instrumentation.psycopg2:
            try:
                Psycopg2Instrumentor().instrument()
                logger.debug("psycopg2 instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument psycopg2: {e}")
        
        if self.config.instrumentation.requests:
            try:
                RequestsInstrumentor().instrument()
                logger.debug("requests instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument requests: {e}")
        
        if self.config.instrumentation.sqlalchemy:
            try:
                SQLAlchemyInstrumentor().instrument()
                logger.debug("SQLAlchemy instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument SQLAlchemy: {e}")
    
    def instrument_fastapi(self, app: Any) -> None:
        """Instrument a FastAPI application."""
        if not self.config.instrumentation.fastapi:
            return
        
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self._tracer_provider,
                excluded_urls=",".join([
                    "/healthz",
                    "/metrics",
                    "/favicon.ico",
                ]),
            )
            logger.debug("FastAPI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")
    
    def init(self) -> None:
        """Initialize OpenTelemetry."""
        if self._initialized:
            logger.warning("OpenTelemetry already initialized")
            return
        
        if not self.is_enabled():
            logger.info("OpenTelemetry is disabled")
            return
        
        logger.info("Initializing OpenTelemetry...")
        
        try:
            # Set up components
            self._setup_tracing()
            self._setup_metrics()
            self._setup_log_correlation()
            self._setup_instrumentation()
            
            self._initialized = True
            logger.info("OpenTelemetry initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            # Clean up partially initialized state
            self.shutdown()
            raise
    
    def shutdown(self) -> None:
        """Shut down OpenTelemetry."""
        if not self._initialized:
            return
        
        logger.info("Shutting down OpenTelemetry...")
        
        # Shut down tracer provider
        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
                logger.debug("Tracer provider shut down")
            except Exception as e:
                logger.error(f"Error shutting down tracer provider: {e}")
        
        # Shut down meter provider
        if self._meter_provider:
            try:
                self._meter_provider.shutdown()
                logger.debug("Meter provider shut down")
            except Exception as e:
                logger.error(f"Error shutting down meter provider: {e}")
        
        # Reset state
        self._tracer_provider = None
        self._meter_provider = None
        self._span_exporters = {}
        self._metric_exporters = {}
        self._initialized = False
        
        logger.info("OpenTelemetry shut down")
    
    @property
    def tracer_provider(self) -> Optional[TracerProvider]:
        """Get the tracer provider."""
        return self._tracer_provider
    
    @property
    def meter_provider(self) -> Optional[MeterProvider]:
        """Get the meter provider."""
        return self._meter_provider
    
    @property
    def tracer(self):
        """Get a tracer for the current service."""
        if not self._tracer_provider:
            return trace.get_tracer(__name__)
        return self._tracer_provider.get_tracer(
            self.config.service_name,
            self.config.service_version,
        )
    
    @property
    def meter(self):
        """Get a meter for the current service."""
        if not self._meter_provider:
            return get_meter_provider().get_meter(__name__)
        return self._meter_provider.get_meter(
            self.config.service_name,
            self.config.service_version,
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global instance
_otel_manager: Optional[OpenTelemetryManager] = None


def get_otel_manager() -> OpenTelemetryManager:
    """Get or create the global OpenTelemetry manager."""
    global _otel_manager
    if _otel_manager is None:
        _otel_manager = OpenTelemetryManager()
    return _otel_manager


def init_opentelemetry(config: Optional[OpenTelemetryConfig] = None) -> OpenTelemetryManager:
    """Initialize OpenTelemetry with the given configuration.
    
    Args:
        config: OpenTelemetry configuration. If None, loads from environment.
    
    Returns:
        The initialized OpenTelemetryManager instance.
    """
    global _otel_manager
    
    if _otel_manager is not None:
        _otel_manager.shutdown()
    
    _otel_manager = OpenTelemetryManager(config)
    _otel_manager.init()
    
    return _otel_manager


def shutdown_opentelemetry() -> None:
    """Shut down OpenTelemetry."""
    global _otel_manager
    
    if _otel_manager is not None:
        _otel_manager.shutdown()
        _otel_manager = None


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI application with OpenTelemetry."""
    manager = get_otel_manager()
    manager.instrument_fastapi(app)


def get_tracer(name: Optional[str] = None):
    """Get a tracer from the global OpenTelemetry manager."""
    if _otel_manager is None:
        return trace.get_tracer(name or __name__)
    return _otel_manager.tracer if name is None else _otel_manager.tracer_provider.get_tracer(name)


def get_meter(name: Optional[str] = None):
    """Get a meter from the global OpenTelemetry manager."""
    if _otel_manager is None:
        return get_meter_provider().get_meter(name or __name__)
    return _otel_manager.meter if name is None else _otel_manager.meter_provider.get_meter(name)


@asynccontextmanager
async def async_trace_span(name: str, **kwargs):
    """Async context manager for creating a span."""
    if _otel_manager is None or not _otel_manager.is_enabled() or not _otel_manager.config.trace.enabled:
        yield
        return
    
    tracer = _otel_manager.tracer
    with tracer.start_as_current_span(name, **kwargs) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def trace_span(name: str, **kwargs):
    """Decorator for creating a span around a function."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kw):
                async with async_trace_span(name, **kwargs):
                    return await func(*args, **kw)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kw):
                with trace.get_tracer(__name__).start_as_current_span(name, **kwargs):
                    return func(*args, **kw)
            return sync_wrapper
    return decorator
