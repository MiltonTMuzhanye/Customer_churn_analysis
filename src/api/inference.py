import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Main class for churn prediction inference"""
    
    def __init__(self, model_path: str = 'models/churn_model.pkl', 
                 preprocessor_path: str = 'models/preprocessor.pkl'):
        """Initialize predictor with model and preprocessor"""
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_model(preprocessor_path)
        self.feature_engineer = None
        
        # Initialize feature engineer
        from features.build_features import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        
        logger.info("ChurnPredictor initialized successfully")
    
    def _load_model(self, path: str) -> Any:
        """Load a pickle model"""
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {path}")
            return model
        except FileNotFoundError:
            logger.error(f"Model not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise
    
    def preprocess_customer(self, customer_data: Dict) -> pd.DataFrame:
        """Preprocess customer data for prediction"""
        try:
            # Convert to DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Apply feature engineering
            if self.feature_engineer:
                customer_df = self.feature_engineer.create_features(customer_df)
            
            return customer_df
            
        except Exception as e:
            logger.error(f"Error preprocessing customer data: {e}")
            raise
    
    def predict_single(self, customer_data: Dict) -> Tuple[int, float]:
        """Predict churn for a single customer"""
        try:
            # Preprocess customer data
            customer_df = self.preprocess_customer(customer_data)
            
            # Make prediction
            prediction = self.model.predict(customer_df)[0]
            probability = self.model.predict_proba(customer_df)[:, 1][0]
            
            logger.info(f"Prediction made: {prediction} with probability {probability:.3f}")
            return prediction, probability
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, customers_data: List[Dict]) -> List[Tuple[int, float]]:
        """Predict churn for multiple customers"""
        try:
            predictions = []
            
            for i, customer_data in enumerate(customers_data):
                try:
                    prediction, probability = self.predict_single(customer_data)
                    predictions.append((prediction, probability))
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(customers_data)} customers")
                        
                except Exception as e:
                    logger.error(f"Error processing customer {i}: {e}")
                    # Append default values for failed predictions
                    predictions.append((-1, 0.0))
            
            logger.info(f"Batch prediction completed: {len(predictions)} predictions made")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_risk_factors(self, customer_data: Dict) -> List[str]:
        """Extract risk factors from customer data"""
        try:
            customer_df = self.preprocess_customer(customer_data)
            customer = customer_df.iloc[0]
            
            risk_factors = []
            
            # Business logic for risk factors
            if customer['Contract'] == 'Month-to-month':
                risk_factors.append("Month-to-month contract (3x higher churn risk)")
            
            if customer['tenure'] < 12:
                risk_factors.append(f"New customer (tenure: {customer['tenure']} months)")
            
            if customer['InternetService'] == 'Fiber optic':
                risk_factors.append("Fiber optic service (higher churn observed)")
            
            if customer['PaperlessBilling'] == 'Yes':
                risk_factors.append("Paperless billing")
            
            if customer['PaymentMethod'] == 'Electronic check':
                risk_factors.append("Electronic check payment")
            
            if 'NumServices' in customer and customer['NumServices'] < 3:
                risk_factors.append(f"Limited services ({customer['NumServices']}/6)")
            
            if 'MonthlyCharges' in customer and customer['MonthlyCharges'] > 70:
                risk_factors.append(f"High monthly charges (${customer['MonthlyCharges']})")
            
            return risk_factors[:5]  # Return top 5 factors
            
        except Exception as e:
            logger.error(f"Error extracting risk factors: {e}")
            return ["Unable to extract risk factors"]
    
    def get_suggested_actions(self, customer_data: Dict, 
                             churn_probability: float) -> List[str]:
        """Generate suggested retention actions"""
        try:
            customer_df = self.preprocess_customer(customer_data)
            customer = customer_df.iloc[0]
            
            actions = []
            
            # Risk-based actions
            if churn_probability > 0.7:
                actions.append("Immediate retention specialist outreach required")
                actions.append("Offer loyalty discount (15-20%)")
            elif churn_probability > 0.4:
                actions.append("Schedule proactive check-in call")
                actions.append("Offer service optimization review")
            
            # Feature-based actions
            if customer['Contract'] == 'Month-to-month':
                actions.append("Convert to annual contract with 15% discount")
            
            if customer['tenure'] < 6:
                actions.append("Enroll in new customer engagement program")
            
            if 'NumServices' in customer and customer['NumServices'] < 3:
                actions.append("Offer bundled service package promotion")
            
            if customer['PaymentMethod'] == 'Electronic check':
                actions.append("Promote automatic payment for $5 monthly discount")
            
            # General actions
            actions.append("Send customer satisfaction survey")
            actions.append("Review recent service tickets")
            
            return actions[:5]  # Return top 5 actions
            
        except Exception as e:
            logger.error(f"Error generating suggested actions: {e}")
            return ["Review customer account manually"]
    
    def get_prediction_summary(self, predictions: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Generate summary statistics for batch predictions"""
        try:
            if not predictions:
                return {"error": "No predictions provided"}
            
            # Extract probabilities
            probabilities = [p[1] for p in predictions]
            churn_predictions = [p[0] for p in predictions]
            
            # Calculate statistics
            total_customers = len(predictions)
            churn_count = sum(churn_predictions)
            avg_probability = np.mean(probabilities)
            
            # Risk segmentation
            high_risk = sum(1 for p in probabilities if p > 0.7)
            medium_risk = sum(1 for p in probabilities if 0.4 <= p <= 0.7)
            low_risk = sum(1 for p in probabilities if p < 0.4)
            
            summary = {
                "total_customers": total_customers,
                "predicted_churn_count": churn_count,
                "predicted_churn_rate": churn_count / total_customers,
                "average_churn_probability": avg_probability,
                "risk_distribution": {
                    "high_risk": high_risk,
                    "medium_risk": medium_risk,
                    "low_risk": low_risk
                },
                "statistics": {
                    "min_probability": min(probabilities),
                    "max_probability": max(probabilities),
                    "std_probability": np.std(probabilities)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating prediction summary: {e}")
            return {"error": str(e)}