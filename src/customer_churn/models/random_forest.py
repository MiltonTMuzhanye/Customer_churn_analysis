# src/customer_churn/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class RandomForestModel:
    """Random Forest model wrapper."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.best_params = None
        
    def create_model(self, **kwargs) -> RandomForestClassifier:
        """Create random forest model."""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 10),
            'min_samples_split': self.config.get('min_samples_split', 5),
            'min_samples_leaf': self.config.get('min_samples_leaf', 2),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': self.config.get('n_jobs', -1)
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        try:
            self.model = self.create_model(**kwargs)
            self.model.fit(X, y)
            logger.info("Random Forest model trained successfully")
            return self.model
        except Exception as e:
            raise ModelTrainingError(f"Random Forest training error: {str(e)}")
    
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