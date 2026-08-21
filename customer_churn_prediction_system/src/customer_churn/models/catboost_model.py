from catboost import CatBoostClassifier
from typing import Dict, Any, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class CatBoostModel:
    """CatBoost model wrapper."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.best_params = None
        
    def create_model(self, **kwargs) -> CatBoostClassifier:
        """Create CatBoost model."""
        params = {
            'iterations': self.config.get('iterations', 100),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'depth': self.config.get('depth', 6),
            'random_seed': self.config.get('random_state', 42),
            'verbose': False
        }
        params.update(kwargs)
        return CatBoostClassifier(**params)
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        try:
            self.model = self.create_model(**kwargs)
            self.model.fit(X, y)
            logger.info("CatBoost model trained successfully")
            return self.model
        except Exception as e:
            raise ModelTrainingError(f"CatBoost training error: {str(e)}")
    
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
        return self.model.get_feature_importance()