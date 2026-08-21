import lightgbm as lgb
from typing import Dict, Any, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class LightGBMModel:
    """LightGBM model wrapper."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.best_params = None
        
    def create_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Create LightGBM model."""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'num_leaves': self.config.get('num_leaves', 31),
            'min_child_samples': self.config.get('min_child_samples', 20),
            'random_state': self.config.get('random_state', 42),
            'verbose': -1
        }
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        try:
            self.model = self.create_model(**kwargs)
            self.model.fit(X, y)
            logger.info("LightGBM model trained successfully")
            return self.model
        except Exception as e:
            raise ModelTrainingError(f"LightGBM training error: {str(e)}")
    
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