# src/customer_churn/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List
from ..utils.logger import default_logger as logger
from ..utils.exceptions import FeatureEngineeringError
from ..utils.config import config_loader

class DataPreprocessor:
    """Handle data preprocessing and transformation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.feature_config = self.config.get("features", {})
        self.target_column = self.config.get("data", {}).get("target_column", "Churn")
        self.id_columns = self.feature_config.get("id_columns", ["customerID"])
        
        self.numerical_features = self.feature_config.get("numerical_features", [])
        self.categorical_features = self.feature_config.get("categorical_features", [])
        
        self.preprocessor = None
        self.scalers = {}
        self.encoders = {}
        self.target_encoder = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data."""
        try:
            df = df.copy()
            
            # Convert TotalCharges to numeric, handling empty strings
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                # Fill missing TotalCharges with Median
                df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
            
            # Convert SeniorCitizen to categorical
            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
                # Add it to categorical features
                if 'SeniorCitizen' not in self.categorical_features:
                    self.categorical_features.append('SeniorCitizen')
            
            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            
            logger.info(f"Data cleaned. Shape: {df.shape}")
            return df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Data cleaning error: {str(e)}")
    
    def create_preprocessor(self, df: pd.DataFrame, fit: bool = True) -> Pipeline:
        """Create a preprocessing pipeline."""
        try:
            # Update features based on actual data
            if not self.numerical_features:
                self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
                self.numerical_features = [col for col in self.numerical_features 
                                         if col not in self.id_columns and col != self.target_column]
            
            if not self.categorical_features:
                self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
                self.categorical_features = [col for col in self.categorical_features 
                                           if col not in self.id_columns and col != self.target_column]
            
            # Create transformers
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ],
                remainder='drop'
            )
            
            if fit:
                # Fit the preprocessor
                X = df[self.numerical_features + self.categorical_features]
                preprocessor.fit(X)
                self.preprocessor = preprocessor
            
            logger.info("Preprocessor created successfully")
            return preprocessor
            
        except Exception as e:
            raise FeatureEngineeringError(f"Preprocessor creation error: {str(e)}")
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the preprocessor."""
        try:
            if self.preprocessor is None:
                self.create_preprocessor(df, fit=True)
            
            # Separate features
            X = df[self.numerical_features + self.categorical_features]
            
            # Transform
            X_transformed = self.preprocessor.transform(X)
            
            # Get feature names
            feature_names = self.get_feature_names()
            
            # Create DataFrame
            X_processed = pd.DataFrame(X_transformed, columns=feature_names)
            
            # Add IDs if present
            for id_col in self.id_columns:
                if id_col in df.columns:
                    X_processed[id_col] = df[id_col].values
            
            logger.info(f"Data transformed. Shape: {X_processed.shape}")
            return X_processed
            
        except Exception as e:
            raise FeatureEngineeringError(f"Data transformation error: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        try:
            if self.preprocessor is None:
                return []
            
            feature_names = []
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(columns))
                    else:
                        # Fallback for older sklearn versions
                        import re
                        for col in columns:
                            unique_values = transformer.categories_[transformer.transformers_.index((name, transformer, columns))]
                            feature_names.extend([f"{col}_{val}" for val in unique_values])
            
            return feature_names
            
        except Exception as e:
            raise FeatureEngineeringError(f"Error getting feature names: {str(e)}")
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable."""
        try:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column {self.target_column} not found")
            
            y = df[self.target_column].copy()
            
            # Encode target if needed
            if y.dtype == 'object':
                if self.target_encoder is None:
                    self.target_encoder = LabelEncoder()
                    y_encoded = self.target_encoder.fit_transform(y)
                else:
                    y_encoded = self.target_encoder.transform(y)
                return pd.Series(y_encoded, name=self.target_column)
            else:
                return y
            
        except Exception as e:
            raise FeatureEngineeringError(f"Target preparation error: {str(e)}")
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform target variable."""
        try:
            if self.target_encoder is not None:
                return self.target_encoder.inverse_transform(y_encoded)
            return y_encoded
        except Exception as e:
            raise FeatureEngineeringError(f"Target inverse transform error: {str(e)}")