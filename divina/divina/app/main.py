""
Main FastAPI application for AthenaMyst:Divina.

Provides health checks, metrics, and API endpoints.
"""
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from typing import Dict, Any

from ..config import settings
from .metrics import metrics_router, record_metrics
from ..logging import logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting AthenaMyst:Divina")
    logger.info(f"Environment: {settings.env.value}")
    logger.info(f"Log level: {settings.log_level}")
    
    # Initialize application state here
    app.state.startup_time = time.time()
    app.state.ready = True
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down AthenaMyst:Divina")
    app.state.ready = False


# Create FastAPI application
app = FastAPI(
    title="AthenaMyst:Divina",
    description="Multi-timeframe FX Signal Engine",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])

# Add middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and record metrics."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Skip logging for health checks
    if request.url.path != "/healthz":
        logger.info(
            "Request processed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )
    
    # Record metrics
    record_metrics(request, response, process_time)
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    logger.warning("Validation error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle all other exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Health check endpoint
@app.get("/healthz", include_in_schema=False)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Kubernetes and load balancers."""
    status = {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.env.value,
        "uptime": time.time() - app.state.startup_time,
        "database": "ok",  # Add database status check if needed
        "oanda": "ok",     # Add OANDA status check if needed
    }
    
    if not app.state.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "starting"},
        )
        
    return status


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic information."""
    return {
        "name": "AthenaMyst:Divina",
        "version": "0.1.0",
        "environment": settings.env.value,
        "docs": "/api/docs",
    }
