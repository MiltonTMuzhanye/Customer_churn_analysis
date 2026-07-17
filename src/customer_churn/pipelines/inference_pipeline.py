# src/customer_churn/pipelines/inference_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..utils.logger import default_logger as logger
from ..utils.config import config_loader
from ..utils.helpers import load_artifact
from ..utils.exceptions import ModelPredictionError

class InferencePipeline:
    """Handle model inference and predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.data_preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer()
        
        self.model = None
        self.preprocessor = None
        self.features = None
        self.threshold = None
        
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load necessary artifacts."""
        try:
            artifacts_dir = Path("artifacts")
            
            # Load model
            model_path = artifacts_dir / "trained_models" / "best_model.pkl"
            if model_path.exists():
                self.model = load_artifact(model_path)
                logger.info("Model loaded successfully")
            
            # Load preprocessor
            preprocessor_path = artifacts_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                self.preprocessor = load_artifact(preprocessor_path)
                self.data_preprocessor.preprocessor = self.preprocessor
                logger.info("Preprocessor loaded successfully")
            
            # Load features
            features_path = artifacts_dir / "feature_lists" / "features.json"
            if features_path.exists():
                self.features = load_artifact(features_path)
                logger.info("Features loaded successfully")
            
            # Load threshold
            threshold_path = artifacts_dir / "threshold_config" / "threshold.json"
            if threshold_path.exists():
                threshold_data = load_artifact(threshold_path)
                self.threshold = threshold_data.get('threshold', 0.5)
                logger.info(f"Threshold loaded: {self.threshold}")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new data."""
        try:
            if self.model is None:
                raise ModelPredictionError("Model not loaded")
            
            # Step 1: Clean data
            df_clean = self.data_preprocessor.clean_data(df)
            
            # Step 2: Feature engineering
            X_engineered = self.feature_engineer.create_features(df_clean)
            
            # Step 3: Preprocess
            if self.preprocessor is not None:
                X_processed = self.preprocessor.transform(X_engineered)
                
                # Get feature names
                feature_names = self.data_preprocessor.get_feature_names()
                if feature_names:
                    X_processed = pd.DataFrame(X_processed, columns=feature_names)
            else:
                # If no preprocessor, use the raw features
                X_processed = X_engineered
            
            # Step 4: Select features
            if self.features:
                X_final = X_processed[self.features]
            else:
                X_final = X_processed
            
            # Step 5: Make predictions
            y_pred = self.model.predict(X_final)
            y_prob = self.model.predict_proba(X_final)[:, 1]
            
            # Step 6: Apply threshold
            y_pred_threshold = (y_prob >= self.threshold).astype(int)
            
            # Step 7: Format results
            results = []
            for idx in range(len(df)):
                result = {
                    'customer_id': df.iloc[idx]['customerID'] if 'customerID' in df.columns else idx,
                    'churn_prediction': int(y_pred_threshold[idx]),
                    'churn_probability': float(y_prob[idx]),
                    'risk_level': self._get_risk_level(y_prob[idx])
                }
                results.append(result)
            
            logger.info(f"Predictions completed for {len(df)} customers")
            return {
                'predictions': results,
                'predicted_labels': y_pred_threshold.tolist(),
                'probabilities': y_prob.tolist()
            }
            
        except Exception as e:
            raise ModelPredictionError(f"Prediction error: {str(e)}")
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level based on probability."""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def batch_predict(self, df: pd.DataFrame, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Make predictions in batches for large datasets."""
        results = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results['predictions'])
        return results