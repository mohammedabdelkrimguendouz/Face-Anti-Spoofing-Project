"""
Main Application Entry Point
----------------------------
FastAPI application for Face Anti-Spoofing service.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import config

# =============================================================
# Logging Setup
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================
# Lifespan Manager
# =============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Face Anti-Spoofing Service...")
    logger.info(f"Configuration: {config}")
    logger.info(f"Models available: {list(config.MODELS.keys())}")
    
    # Pre-load default model
    try:
        from backend.services.inference_service import inference_service
        inference_service.load_model(config.DEFAULT_MODEL)
        logger.info(f"Pre-loaded default model: {config.DEFAULT_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to pre-load default model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Anti-Spoofing Service...")


# =============================================================
# Create FastAPI App
# =============================================================
app = FastAPI(
    title="Face Anti-Spoofing API",
    description="""
    API for detecting face spoofing attacks using deep learning models.
    
    ## Models Available:
    - **LSTM Autoencoder**: Undercomplete autoencoder with LSTM
    - **BiLSTM VAE**: Variational autoencoder with bidirectional LSTM
    
    ## Features:
    - Real-time inference
    - Batch processing
    - Multiple model support
    - Configurable thresholds
    """,
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================
# CORS Middleware
# =============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Register Routes
# =============================================================
app.include_router(router, prefix="/api/v1")


# =============================================================
# Root Endpoint
# =============================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Face Anti-Spoofing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "models": "/api/v1/models"
    }


# =============================================================
# Run Server
# =============================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG
    )