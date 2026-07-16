from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import custom modules
from features.build_features import FeatureEngineer

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn risk",
    version="1.0.0"
)

# Load model and preprocessor
MODEL_PATH = 'models/churn_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️  Model not found at {MODEL_PATH}")
    model = None

try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"✅ Preprocessor loaded from {PREPROCESSOR_PATH}")
except FileNotFoundError:
    print(f"⚠️  Preprocessor not found at {PREPROCESSOR_PATH}")
    preprocessor = None

# Pydantic models
class CustomerData(BaseModel):
    """Schema for customer data input"""
    customerID: Optional[str] = None
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
    """Schema for prediction response"""
    customer_id: Optional[str]
    churn_prediction: str
    churn_probability: float
    risk_score: str
    confidence: float
    timestamp: str
    risk_factors: List[str]
    suggested_actions: List[str]

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    customers: List[CustomerData]

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict_batch", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn for a single customer"""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Please check server configuration."
        )
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        customer_df = pd.DataFrame([customer_dict])
        
        # Apply feature engineering
        engineer = FeatureEngineer()
        customer_df = engineer.create_features(customer_df)
        
        # Make prediction
        prediction = model.predict(customer_df)
        probability = model.predict_proba(customer_df)[:, 1][0]
        
        # Determine risk score
        if probability > 0.7:
            risk_score = "High"
        elif probability > 0.4:
            risk_score = "Medium"
        else:
            risk_score = "Low"
        
        # Generate risk factors
        risk_factors = _get_risk_factors(customer_df.iloc[0])
        
        # Generate suggested actions
        suggested_actions = _get_suggested_actions(customer_df.iloc[0], risk_score)
        
        response = PredictionResponse(
            customer_id=customer.customerID,
            churn_prediction="Yes" if prediction[0] == 1 else "No",
            churn_probability=float(probability),
            risk_score=risk_score,
            confidence=float(probability) if prediction[0] == 1 else float(1 - probability),
            timestamp=datetime.now().isoformat(),
            risk_factors=risk_factors,
            suggested_actions=suggested_actions
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded"
        )
    
    try:
        predictions = []
        churn_count = 0
        total_customers = len(batch_request.customers)
        
        for customer in batch_request.customers:
            # Predict for each customer
            single_prediction = await predict_churn(customer)
            predictions.append(single_prediction)
            
            if single_prediction.churn_prediction == "Yes":
                churn_count += 1
        
        # Create summary
        summary = {
            "total_customers": total_customers,
            "churn_count": churn_count,
            "churn_rate": churn_count / total_customers if total_customers > 0 else 0,
            "average_risk_score": np.mean([p.churn_probability for p in predictions]),
            "high_risk_customers": len([p for p in predictions if p.risk_score == "High"]),
            "medium_risk_customers": len([p for p in predictions if p.risk_score == "Medium"]),
            "low_risk_customers": len([p for p in predictions if p.risk_score == "Low"])
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_risk_factors(customer: pd.Series) -> List[str]:
    """Extract risk factors from customer data"""
    risk_factors = []
    
    # Business logic for risk factors
    if customer['Contract'] == 'Month-to-month':
        risk_factors.append("Month-to-month contract")
    
    if customer['tenure'] < 12:
        risk_factors.append("New customer (< 1 year)")
    
    if customer['InternetService'] == 'Fiber optic':
        risk_factors.append("Fiber optic service (higher churn observed)")
    
    if customer['PaperlessBilling'] == 'Yes':
        risk_factors.append("Paperless billing")
    
    if customer['PaymentMethod'] == 'Electronic check':
        risk_factors.append("Electronic check payment")
    
    if 'NumServices' in customer and customer['NumServices'] < 3:
        risk_factors.append(f"Limited services ({customer['NumServices']})")
    
    return risk_factors[:5]  # Return top 5 factors

def _get_suggested_actions(customer: pd.Series, risk_score: str) -> List[str]:
    """Generate suggested retention actions"""
    actions = []
    
    if risk_score == "High":
        actions.append("Immediate retention call required")
        actions.append("Offer loyalty discount (15-20%)")
    
    if customer['Contract'] == 'Month-to-month':
        actions.append("Convert to annual contract with 15% discount")
    
    if customer['tenure'] < 6:
        actions.append("Engage with new customer onboarding program")
    
    if 'NumServices' in customer and customer['NumServices'] < 3:
        actions.append("Offer bundled service package")
    
    if customer['PaymentMethod'] == 'Electronic check':
        actions.append("Promote automatic payment methods")
    
    actions.append("Review customer satisfaction survey results")
    
    return actions[:5]  # Return top 5 actions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)