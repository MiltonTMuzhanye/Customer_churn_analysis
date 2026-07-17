from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .schemas import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from ..inference.predictor import Predictor
from src.customer_churn.utils.logger import default_logger as logger

router = APIRouter()

# Initialize predictor
predictor = Predictor()

@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Predict churn for a single customer."""
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.dict()])
        
        # Make prediction
        result = predictor.predict(df)
        
        # Format response
        prediction = result['predictions'][0]
        response = PredictionResponse(
            customer_id=prediction['customer_id'],
            churn_prediction=bool(prediction['churn_prediction']),
            churn_probability=prediction['churn_probability'],
            risk_level=prediction['risk_level']
        )
        
        logger.info(f"Prediction made for customer {request.customerID}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_churn(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    try:
        # Convert request to DataFrame
        df = pd.DataFrame(request.customers)
        
        # Make predictions
        results = predictor.predict(df)
        
        # Format response
        predictions = []
        for pred in results['predictions']:
            predictions.append({
                'customer_id': pred['customer_id'],
                'churn_prediction': bool(pred['churn_prediction']),
                'churn_probability': pred['churn_probability'],
                'risk_level': pred['risk_level']
            })
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=sum(1 for p in predictions if p['churn_prediction'])
        )
        
        logger.info(f"Batch prediction made for {len(predictions)} customers")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for model updates)."""
    try:
        background_tasks.add_task(predictor.load_model)
        return {"message": "Model reload started in background"}
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )