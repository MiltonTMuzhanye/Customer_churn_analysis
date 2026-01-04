import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple

class FeaturePipeline:
    """Complete feature preprocessing pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.cat_cols = None
        self.num_cols = None
        
    def fit(self, X: pd.DataFrame):
        """Fit the preprocessing pipeline"""
        # Identify categorical and numeric columns
        self.cat_cols = X.select_dtypes(include='object').columns.tolist()
        self.num_cols = X.select_dtypes(include='number').columns.tolist()
        
        # Remove engineered features from automatic detection
        engineered_features = ['NumServices', 'ChargeRatio', 'HighValue', 'ContractMonths', 'MonthlyValue']
        self.num_cols = [col for col in self.num_cols if col not in engineered_features]
        
        # Add engineered features to appropriate lists
        for feature in engineered_features:
            if feature in X.columns:
                if X[feature].dtype == 'object':
                    self.cat_cols.append(feature)
                else:
                    self.num_cols.append(feature)
        
        # Define preprocessing pipelines
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])
        
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline"""
        if self.preprocessor is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            raise ValueError("Pipeline not fitted yet.")
        
        # Get numeric feature names
        num_features = self.num_cols
        
        # Get categorical feature names
        if 'cat' in self.preprocessor.named_transformers_:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_features = cat_encoder.get_feature_names_out(self.cat_cols)
        else:
            cat_features = []
        
        return list(num_features) + list(cat_features)