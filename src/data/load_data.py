import pandas as pd
import numpy as np
from typing import Dict, Tuple
import json

class DataLoader:
    """Load and validate raw data"""
    
    @staticmethod
    def load_raw_data(filepath: str = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
        """Load raw customer data"""
        print(f"📥 Loading raw data from {filepath}")
        data = pd.read_csv(filepath)
        print(f"✅ Loaded {len(data)} records with {len(data.columns)} features")
        return data
    
    @staticmethod
    def validate_data_schema(data: pd.DataFrame) -> Dict:
        """Validate data schema and quality"""
        validation_results = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'basic_stats': {}
        }
        
        # Check for required columns
        required_cols = ['customerID', 'Churn']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            validation_results['basic_stats'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        return validation_results
    
    @staticmethod
    def save_validation_report(validation_results: Dict, filepath: str = 'data/raw/validation_report.json'):
        """Save validation report"""
        with open(filepath, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"💾 Validation report saved to {filepath}")