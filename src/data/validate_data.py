import pandas as pd
import numpy as np
from typing import Dict, List

class DataValidator:
    """Data validation and quality checks"""
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality checks"""
        checks = {
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'unique_counts': {},
            'outliers': {}
        }
        
        # Unique counts for categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            checks['unique_counts'][col] = data[col].nunique()
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            checks['outliers'][col] = outliers
        
        return checks
    
    @staticmethod
    def validate_target_variable(data: pd.DataFrame, target_col: str = 'Churn') -> Dict:
        """Validate target variable distribution"""
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        target_stats = {
            'distribution': data[target_col].value_counts().to_dict(),
            'percentage': (data[target_col].value_counts(normalize=True) * 100).to_dict(),
            'imbalance_ratio': None
        }
        
        # Calculate imbalance ratio
        class_counts = data[target_col].value_counts()
        if len(class_counts) == 2:
            target_stats['imbalance_ratio'] = max(class_counts) / min(class_counts)
        
        return target_stats