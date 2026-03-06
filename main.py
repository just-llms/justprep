"""FastAPI application entry point."""

import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from util.logger import setup_logging
from routes.websocket_routes import router as websocket_router
from routes.resume_routes import router as resume_router
from routes.transcript_routes import router as transcript_router

# Load environment variables
load_dotenv()

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_format = os.getenv("LOG_FORMAT", "standard")
setup_logging(log_level=log_level, log_format=log_format)

# Get logger for this module
import logging
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Interview System",
    description="Voice-based AI mock interview engine",
    version="0.1.0"
)

# CORS middleware (supports WebSocket connections)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(websocket_router)
app.include_router(resume_router)
app.include_router(transcript_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Application startup event handler."""
    logger.info("Starting AI Interview System")
    logger.info(f"Log level: {log_level}, Format: {log_format}")
    
    # Start system metrics collection
    try:
        from util.metrics import MetricsCollector
        metrics = MetricsCollector.get_instance()
        metrics.start_system_metrics_collection(interval_seconds=10)
        logger.info("System metrics collection started")
    except Exception as e:
        logger.warning(f"Failed to start system metrics collection: {e}")
    
    # Start stuck session detection
    try:
        from core.safety import SafetyController
        safety_controller = SafetyController.get_instance()
        safety_controller.start_stuck_session_detection()
        logger.info("Stuck session detection started")
    except Exception as e:
        logger.warning(f"Failed to start stuck session detection: {e}")
    
    # Start periodic metrics logging
    async def log_metrics_periodically() -> None:
        """Log metrics every 60 seconds."""
        import asyncio
        from util.metrics import MetricsCollector
        from util.logger import get_component_logger
        
        metrics_logger = get_component_logger("metrics")
        metrics = MetricsCollector.get_instance()
        
        while True:
            await asyncio.sleep(60)  # Log every 60 seconds
            try:
                exported = metrics.export_metrics()
                metrics_logger.info(
                    "metrics_export",
                    "Periodic metrics export",
                    exported
                )
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
    
    asyncio.create_task(log_metrics_periodically())
    logger.info("Periodic metrics logging started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Application shutdown event handler."""
    logger.info("Shutting down AI Interview System")
    
    # Stop system metrics collection
    try:
        from util.metrics import MetricsCollector
        metrics = MetricsCollector.get_instance()
        metrics.stop_system_metrics_collection()
        logger.info("System metrics collection stopped")
    except Exception as e:
        logger.warning(f"Error stopping system metrics collection: {e}")
    
    # Stop stuck session detection
    try:
        from core.safety import SafetyController
        safety_controller = SafetyController.get_instance()
        safety_controller.stop_stuck_session_detection()
        logger.info("Stuck session detection stopped")
    except Exception as e:
        logger.warning(f"Error stopping stuck session detection: {e}")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.
    
    Returns:
        Dictionary with status information
    """
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.
    
    Returns:
        Dictionary with API information
    """
    return {
        "message": "AI Interview System API",
        "version": "0.1.0",
        "docs": "/docs"
    }

