# src/customer_churn/features/engineering.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.logger import default_logger as logger
from ..utils.exceptions import FeatureEngineeringError

class FeatureEngineer:
    """Handle feature engineering and creation."""
    
    def __init__(self):
        self.created_features = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data."""
        try:
            df = df.copy()
            
            # Create tenure categories
            if 'tenure' in df.columns:
                df['tenure_category'] = pd.cut(
                    df['tenure'],
                    bins=[-1, 0, 12, 24, 48, 72],
                    labels=['New', 'Short-term', 'Medium-term', 'Long-term', 'Very-long-term']
                )
                self.created_features.append('tenure_category')
            
            # Create average monthly charges
            if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
                df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
                self.created_features.append('avg_monthly_charges')
            
            # Create service count
            service_columns = [
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            available_services = [col for col in service_columns if col in df.columns]
            if available_services:
                df['service_count'] = df[available_services].apply(
                    lambda x: (x != 'No').sum() + (x == 'Yes').sum(), axis=1
                )
                self.created_features.append('service_count')
            
            # Create feature interaction: tenure * monthly charges
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                df['tenure_monthly_charges'] = df['tenure'] * df['MonthlyCharges']
                self.created_features.append('tenure_monthly_charges')
            
            # Create contract type indicator
            if 'Contract' in df.columns:
                df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
                self.created_features.append('is_month_to_month')
                
                df['is_one_year'] = (df['Contract'] == 'One year').astype(int)
                self.created_features.append('is_one_year')
                
                df['is_two_year'] = (df['Contract'] == 'Two year').astype(int)
                self.created_features.append('is_two_year')
            
            # Create payment method indicators
            if 'PaymentMethod' in df.columns:
                df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
                self.created_features.append('is_electronic_check')
            
            # Create customer value score
            if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
                # Normalize and combine
                monthly_norm = df['MonthlyCharges'] / df['MonthlyCharges'].max()
                tenure_norm = df['tenure'] / df['tenure'].max()
                df['customer_value_score'] = monthly_norm * 0.6 + tenure_norm * 0.4
                self.created_features.append('customer_value_score')
            
            # Create risk score
            if all(col in df.columns for col in ['tenure', 'Contract', 'PaperlessBilling']):
                risk_factors = pd.DataFrame()
                risk_factors['tenure_risk'] = (df['tenure'] < 12).astype(int)
                risk_factors['contract_risk'] = (df['Contract'] == 'Month-to-month').astype(int)
                risk_factors['paperless_risk'] = (df['PaperlessBilling'] == 'Yes').astype(int)
                df['risk_score'] = risk_factors.sum(axis=1) / 3
                self.created_features.append('risk_score')
            
            logger.info(f"Created {len(self.created_features)} new features")
            return df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature creation error: {str(e)}")
    
    def get_feature_importance_ranking(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance rankings."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                raise ValueError("Model does not have feature_importances_ or coef_")
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature importance calculation error: {str(e)}")