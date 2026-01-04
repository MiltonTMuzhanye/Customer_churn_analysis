import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app, CustomerData, PredictionResponse

# Create test client
client = TestClient(app)

class TestAPI:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Customer Churn Prediction API"
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
    
    def test_predict_endpoint_valid(self):
        """Test predict endpoint with valid data"""
        # Valid customer data
        customer_data = {
            "customerID": "test-123",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
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
        
        response = client.post("/predict", json=customer_data)
        
        # Check response
        assert response.status_code == 200 or response.status_code == 503
        # 200 if model loaded, 503 if not
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert "risk_score" in data
            assert "risk_factors" in data
            assert "suggested_actions" in data
            
            # Check data types
            assert isinstance(data["churn_prediction"], str)
            assert isinstance(data["churn_probability"], float)
            assert 0 <= data["churn_probability"] <= 1
            assert isinstance(data["risk_factors"], list)
            assert isinstance(data["suggested_actions"], list)
    
    def test_predict_endpoint_invalid(self):
        """Test predict endpoint with invalid data"""
        # Invalid customer data (missing required fields)
        invalid_data = {
            "gender": "Female",
            "SeniorCitizen": 0
            # Missing other required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 Unprocessable Entity
        assert response.status_code == 422
    
    def test_predict_endpoint_invalid_types(self):
        """Test predict endpoint with invalid data types"""
        # Customer data with invalid types
        invalid_data = {
            "customerID": 123,  # Should be string
            "gender": "Female",
            "SeniorCitizen": "yes",  # Should be integer
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": "twelve",  # Should be integer
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": "seventy",  # Should be float
            "TotalCharges": 1000.0
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 Unprocessable Entity
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self):
        """Test batch predict endpoint"""
        # Create batch request with two customers
        batch_data = {
            "customers": [
                {
                    "customerID": "test-1",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
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
                },
                {
                    "customerID": "test-2",
                    "gender": "Male",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "Yes",
                    "tenure": 48,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": 120.0,
                    "TotalCharges": 5000.0
                }
            ]
        }
        
        response = client.post("/predict_batch", json=batch_data)
        
        # Check response
        assert response.status_code == 200 or response.status_code == 503
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "summary" in data
            
            # Check predictions
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == 2
            
            # Check summary
            summary = data["summary"]
            assert "total_customers" in summary
            assert "churn_count" in summary
            assert "churn_rate" in summary
            assert "average_risk_score" in summary
    
    def test_batch_predict_empty(self):
        """Test batch predict with empty list"""
        batch_data = {
            "customers": []  # Empty list
        }
        
        response = client.post("/predict_batch", json=batch_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["summary"]["total_customers"] == 0
            assert data["summary"]["churn_rate"] == 0
    
    def test_pydantic_models(self):
        """Test Pydantic models validation"""
        # Test CustomerData model
        customer = CustomerData(
            customerID="test-123",
            gender="Female",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=12,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="No",
            OnlineBackup="No",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=70.0,
            TotalCharges=1000.0
        )
        
        assert customer.customerID == "test-123"
        assert customer.gender == "Female"
        assert customer.SeniorCitizen == 0
        assert customer.MonthlyCharges == 70.0
        
        # Test with optional customerID
        customer2 = CustomerData(
            gender="Male",
            SeniorCitizen=1,
            Partner="No",
            Dependents="Yes",
            tenure=24,
            PhoneService="Yes",
            MultipleLines="Yes",
            InternetService="Fiber optic",
            OnlineSecurity="Yes",
            OnlineBackup="Yes",
            DeviceProtection="Yes",
            TechSupport="Yes",
            StreamingTV="Yes",
            StreamingMovies="Yes",
            Contract="One year",
            PaperlessBilling="No",
            PaymentMethod="Bank transfer (automatic)",
            MonthlyCharges=100.0,
            TotalCharges=2000.0
        )
        
        assert customer2.customerID is None
        assert customer2.gender == "Male"
        assert customer2.Contract == "One year"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])