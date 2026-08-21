from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class LogisticRegressionModel:
    """Logistic Regression model wrapper."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.best_params = None
        
    def create_model(self, **kwargs) -> LogisticRegression:
        """Create logistic regression model."""
        params = {
            'C': self.config.get('C', 1.0),
            'max_iter': self.config.get('max_iter', 1000),
            'solver': self.config.get('solver', 'liblinear'),
            'random_state': self.config.get('random_state', 42)
        }
        params.update(kwargs)
        return LogisticRegression(**params)
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        try:
            self.model = self.create_model(**kwargs)
            self.model.fit(X, y)
            logger.info("Logistic Regression model trained successfully")
            return self.model
        except Exception as e:
            raise ModelTrainingError(f"Logistic Regression training error: {str(e)}")
    
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
    
    def get_params(self):
        """Get model parameters."""
        if self.model is None:
            return {}
        return self.model.get_params()