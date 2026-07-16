# src/customer_churn/training/trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer
from typing import Dict, Any, Tuple, Optional
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError
from ..utils.config import config_loader
from ..utils.helpers import save_artifact, load_artifact, get_timestamp

class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.model_config = self.config.get("model", {})
        self.training_config = self.config.get("training", {})
        self.evaluation_config = self.config.get("evaluation", {})
        
        self.model = None
        self.model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        
        # Setup MLflow
        mlflow.set_tracking_uri("mlflow/")
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        test_size = self.config.get("data", {}).get("test_size", 0.2)
        random_state = self.config.get("data", {}).get("random_state", 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_class, model_name: str, X_train: pd.DataFrame = None, 
                   y_train: pd.Series = None, **kwargs) -> Any:
        """Train a model with given parameters."""
        try:
            X_train = X_train if X_train is not None else self.X_train
            y_train = y_train if y_train is not None else self.y_train
            
            if X_train is None or y_train is None:
                raise ModelTrainingError("Training data not prepared")
            
            # Initialize model
            model_params = self.model_config.get("models", {}).get(model_name, {})
            model_params.update(kwargs)
            
            model_instance = model_class(model_params)
            model = model_instance.train(X_train, y_train)
            
            self.model = model
            self.model_name = model_name
            
            logger.info(f"Trained {model_name} model successfully")
            return model
            
        except Exception as e:
            raise ModelTrainingError(f"Model training error: {str(e)}")
    
    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, cv: int = None) -> Dict[str, float]:
        """Perform cross-validation."""
        cv = cv or self.model_config.get("cv_folds", 5)
        scoring_metric = self.model_config.get("scoring_metric", "roc_auc")
        
        # Map scoring metric to sklearn scorer
        scoring_map = {
            'roc_auc': 'roc_auc',
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'log_loss': 'neg_log_loss'
        }
        
        scoring = scoring_map.get(scoring_metric, 'roc_auc')
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            'cv_folds': cv,
            'scoring_metric': scoring_metric
        }
        
        logger.info(f"Cross-validation results: {results}")
        return results
    
    def save_model(self, model_path: str = None, metadata: Dict[str, Any] = None):
        """Save trained model and metadata."""
        try:
            model_path = model_path or self.training_config.get("model_save_path")
            if not model_path:
                raise ModelTrainingError("No model save path provided")
            
            # Create timestamp
            timestamp = get_timestamp()
            
            # Save model
            model_file = Path(model_path) / f"{self.model_name}_{timestamp}.pkl"
            save_artifact(self.model, model_file)
            
            # Save metadata
            metadata_file = Path(model_path) / f"{self.model_name}_{timestamp}_metadata.json"
            metadata = metadata or {}
            metadata.update({
                'model_name': self.model_name,
                'timestamp': timestamp,
                'train_shape': self.X_train.shape if self.X_train is not None else None,
                'test_shape': self.X_test.shape if self.X_test is not None else None,
                'params': self.model.get_params() if hasattr(self.model, 'get_params') else {},
                'features': list(self.X_train.columns) if self.X_train is not None else []
            })
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log with MLflow
            with mlflow.start_run(run_name=f"{self.model_name}_{timestamp}"):
                mlflow.log_params(metadata.get('params', {}))
                mlflow.log_metrics(metadata.get('metrics', {}))
                mlflow.sklearn.log_model(self.model, self.model_name)
            
            logger.info(f"Model and metadata saved to {model_path}")
            return model_file
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path: str) -> Any:
        """Load saved model."""
        try:
            model = load_artifact(model_path)
            self.model = model
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            raise ModelTrainingError(f"Error loading model: {str(e)}")