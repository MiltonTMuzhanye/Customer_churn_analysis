import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
from src.customer_churn.pipelines.inference_pipeline import InferencePipeline
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

class Predictor:
    """Wrapper for inference pipeline."""
    
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.model_info = None
        
    def load_model(self):
        """Load the model and artifacts."""
        try:
            self.pipeline = InferencePipeline()
            self.model = self.pipeline.model
            self.model_info = self._get_model_info()
            logger.info("Model loaded successfully in predictor")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions."""
        if self.pipeline is None:
            self.load_model()
        
        return self.pipeline.predict(df)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            # Load model metadata
            artifacts_dir = Path("artifacts")
            metadata_path = artifacts_dir / "trained_models" / "model_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            
            return {
                'model_name': 'customer_churn_model',
                'model_version': '1.0.0',
                'features': self.pipeline.features if self.pipeline else [],
                'threshold': self.pipeline.threshold if self.pipeline else 0.5,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model_info is None:
            self.model_info = self._get_model_info()
        return self.model_info