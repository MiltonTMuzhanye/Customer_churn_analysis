from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"

class InternetService(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"

class ContractType(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"

class PaymentMethod(str, Enum):
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"

class CustomerData(BaseModel):
    """Schema for customer data input"""
    customerID: Optional[str] = Field(None, description="Customer identifier")
    gender: Gender = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0=No, 1=Yes)")
    Partner: YesNo = Field(..., description="Has partner")
    Dependents: YesNo = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, le=100, description="Tenure in months")
    PhoneService: YesNo = Field(..., description="Has phone service")
    MultipleLines: str = Field(..., description="Multiple lines (Yes/No/No phone service)")
    InternetService: InternetService = Field(..., description="Type of internet service")
    OnlineSecurity: str = Field(..., description="Online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Streaming movies (Yes/No/No internet service)")
    Contract: ContractType = Field(..., description="Contract type")
    PaperlessBilling: YesNo = Field(..., description="Paperless billing")
    PaymentMethod: PaymentMethod = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "1234-ABCD",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 1000.0
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    customer_id: Optional[str]
    churn_prediction: str = Field(..., description="Churn prediction (Yes/No)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_score: RiskLevel = Field(..., description="Risk level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    risk_factors: List[str] = Field(..., description="Key risk factors identified")
    suggested_actions: List[str] = Field(..., description="Suggested retention actions")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "1234-ABCD",
                "churn_prediction": "Yes",
                "churn_probability": 0.85,
                "risk_score": "High",
                "confidence": 0.85,
                "timestamp": "2024-01-15T10:30:00Z",
                "risk_factors": [
                    "Month-to-month contract",
                    "New customer (< 1 year)",
                    "Fiber optic service"
                ],
                "suggested_actions": [
                    "Offer annual contract with 15% discount",
                    "Assign retention specialist",
                    "Review service package"
                ]
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    customers: List[CustomerData] = Field(..., description="List of customers to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    CustomerData.Config.schema_extra["example"],
                    CustomerData.Config.schema_extra["example"]
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    PredictionResponse.Config.schema_extra["example"],
                    PredictionResponse.Config.schema_extra["example"]
                ],
                "summary": {
                    "total_customers": 2,
                    "churn_count": 1,
                    "churn_rate": 0.5,
                    "average_risk_score": 0.65
                }
            }
        }

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    model_loaded: bool = Field(..., description="Is model loaded")
    preprocessor_loaded: bool = Field(..., description="Is preprocessor loaded")
    version: str = Field(..., description="API version")

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    detail: str = Field(..., description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())