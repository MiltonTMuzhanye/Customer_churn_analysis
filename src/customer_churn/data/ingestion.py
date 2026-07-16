import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from ..utils.logger import default_logger as logger
from ..utils.exceptions import DataIngestionError
from ..utils.config import config_loader

class DataIngestor:
    """Handle data ingestion and loading."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.data_config = self.config.get("data", {})
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            file_path = file_path or self.data_config.get("raw_path")
            if not file_path:
                raise DataIngestionError("No file path provided")
            
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            raise DataIngestionError(f"File not found: {file_path}")
        except Exception as e:
            raise DataIngestionError(f"Error loading data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data structure and types."""
        try:
            # Check for required columns
            expected_columns = [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ]
            
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                raise DataIngestionError(f"Missing columns: {missing_columns}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"Null values found: {null_counts[null_counts > 0]}")
            
            return True
            
        except Exception as e:
            raise DataIngestionError(f"Data validation failed: {str(e)}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information."""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum()
        }
        return info

    def save_processed_data(self, df: pd.DataFrame, output_path: str = None):
        """Save processed data to disk."""
        try:
            output_path = output_path or self.data_config.get("processed_path")
            if not output_path:
                raise DataIngestionError("No output path provided")
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            raise DataIngestionError(f"Error saving data: {str(e)}")