import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        print("🔧 Creating features...")
        
        features_data = data.copy()
        
        # 1. Create tenure groups
        features_data['TenureGroup'] = pd.cut(
            features_data['tenure'], 
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
        )
        
        # 2. Create service count feature
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Only use columns that exist
        existing_service_cols = [col for col in service_cols if col in features_data.columns]
        features_data['NumServices'] = (features_data[existing_service_cols] == 'Yes').sum(axis=1)
        
        # 3. Create charge ratio feature
        features_data['ChargeRatio'] = features_data['MonthlyCharges'] / (features_data['TotalCharges'] + 1e-6)
        
        # 4. Create high-value customer flag
        features_data['HighValue'] = (
            (features_data['MonthlyCharges'] > features_data['MonthlyCharges'].quantile(0.75)) &
            (features_data['tenure'] > 12)
        ).astype(int)
        
        # 5. Create contract-months interaction
        contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        features_data['ContractMonths'] = features_data['Contract'].map(contract_mapping)
        features_data['MonthlyValue'] = features_data['MonthlyCharges'] * features_data['ContractMonths']
        
        print(f"✅ Features created. Total features: {len(features_data.columns)}")
        return features_data
    
    @staticmethod
    def save_features(data: pd.DataFrame, filepath: str = 'data/features/features.parquet'):
        """Save engineered features"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_parquet(filepath, index=False)
        print(f"💾 Features saved to {filepath}")
    
    @staticmethod
    def get_feature_description() -> Dict:
        """Return description of engineered features"""
        return {
            'TenureGroup': 'Customer tenure categorized into time periods',
            'NumServices': 'Count of subscribed services',
            'ChargeRatio': 'Ratio of monthly to total charges',
            'HighValue': 'Flag for high-value customers',
            'ContractMonths': 'Contract duration in months',
            'MonthlyValue': 'Monthly charges weighted by contract duration'
        }