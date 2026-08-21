import xgboost as xgb
from typing import Dict, Any, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class XGBoostModel:
    """XGBoost model wrapper."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.best_params = None
        
    def create_model(self, **kwargs) -> xgb.XGBClassifier:
        """Create XGBoost model."""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'max_depth': self.config.get('max_depth', 6),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        try:
            self.model = self.create_model(**kwargs)
            self.model.fit(X, y)
            logger.info("XGBoost model trained successfully")
            return self.model
        except Exception as e:
            raise ModelTrainingError(f"XGBoost training error: {str(e)}")
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ModelTrainingError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ModelTrainingError("Model not trained")
        return self.model.predict_proba(X)
    
    def get_feature_importances(self):
        """Get feature importances."""
        if self.model is None:
            return None
        return self.model.feature_importances_