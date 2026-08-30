from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any
import pandas as pd
from .routes import router
from ..inference.predictor import Predictor
from ..inference.preprocess import DataPreprocessor
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = Predictor()
data_preprocessor = DataPreprocessor()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting up API...")
    try:
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down API...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }

# Include routers
app.include_router(router, prefix="/api/v1", tags=["predictions"])

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )