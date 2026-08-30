from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    customer_id: str
    churn_prediction: bool
    churn_probability: float
    risk_level: str  # Low, Medium, High

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    customers: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[Dict[str, Any]]
    total_customers: int
    churn_count: int

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"

class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_name: str
    model_version: str
    features: List[str]
    threshold: float
    metrics: Dict[str, float]
    training_date: str
    last_updated: str