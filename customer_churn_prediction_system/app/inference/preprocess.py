import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.customer_churn.data.preprocessing import DataPreprocessor
from src.customer_churn.features.engineering import FeatureEngineer
from src.customer_churn.utils.logger import default_logger as logger

class DataPreprocessor:
    """Data preprocessing for inference."""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        
    def load_preprocessor(self):
        """Load the preprocessor from artifacts."""
        try:
            from src.customer_churn.utils.helpers import load_artifact
            self.preprocessor = load_artifact("artifacts/preprocessor.pkl")
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for inference."""
        try:
            if self.preprocessor is None:
                self.load_preprocessor()
            
            # Clean data
            df_clean = self._clean_data(df)
            
            # Feature engineering
            X_engineered = self.feature_engineer.create_features(df_clean)
            
            # Transform
            X_transformed = self.preprocessor.transform(X_engineered)
            
            # Get feature names
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback for older sklearn versions
                feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
            
            return pd.DataFrame(X_transformed, columns=feature_names)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for preprocessing."""
        df = df.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Convert TotalCharges to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Convert SeniorCitizen to string
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        
        return df