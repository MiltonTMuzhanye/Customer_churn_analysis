import pandas as pd
import numpy as np
from typing import Dict

class DataCleaner:
    """Clean and preprocess raw data"""
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Main data cleaning pipeline"""
        print("🧹 Cleaning data...")
        
        # Create a copy
        cleaned_data = data.copy()
        
        # 1. Remove customerID (not useful for modeling)
        if 'customerID' in cleaned_data.columns:
            cleaned_data.drop('customerID', axis=1, inplace=True)
        
        # 2. Convert TotalCharges to numeric
        cleaned_data['TotalCharges'] = pd.to_numeric(
            cleaned_data['TotalCharges'], 
            errors='coerce'
        )
        
        # 3. Fill missing TotalCharges with median
        cleaned_data['TotalCharges'].fillna(
            cleaned_data['TotalCharges'].median(), 
            inplace=True
        )
        
        # 4. Standardize categorical values
        service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_cols:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].replace('No internet service', 'No')
        
        print(f"✅ Data cleaned. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    @staticmethod
    def save_clean_data(data: pd.DataFrame, filepath: str = 'data/processed/cleaned_customers.csv'):
        """Save cleaned data"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"💾 Cleaned data saved to {filepath}")