import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specified columns from DataFrame."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""
    
    def __init__(self, date_column: str, features: List[str] = ['year', 'month', 'day', 'dayofweek']):
        self.date_column = date_column
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.date_column in X.columns:
            X[self.date_column] = pd.to_datetime(X[self.date_column])
            
            if 'year' in self.features:
                X['year'] = X[self.date_column].dt.year
            if 'month' in self.features:
                X['month'] = X[self.date_column].dt.month
            if 'day' in self.features:
                X['day'] = X[self.date_column].dt.day
            if 'dayofweek' in self.features:
                X['dayofweek'] = X[self.date_column].dt.dayofweek
                
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers using IQR method."""
    
    def __init__(self, columns: List[str], threshold: float = 1.5):
        self.columns = columns
        self.threshold = threshold
        
    def fit(self, X, y=None):
        self.lower_bounds = {}
        self.upper_bounds = {}
        
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds[col] = Q1 - self.threshold * IQR
                self.upper_bounds[col] = Q3 + self.threshold * IQR
                
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                mask = (X[col] >= self.lower_bounds[col]) & (X[col] <= self.upper_bounds[col])
                X = X[mask]
        return X