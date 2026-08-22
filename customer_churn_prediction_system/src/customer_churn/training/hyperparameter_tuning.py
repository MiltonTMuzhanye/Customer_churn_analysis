from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
from typing import Dict, Any, List, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError
from ..utils.config import config_loader

class HyperparameterTuner:
    """Handle hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("model")
        self.tuning_config = self.config.get("hyperparameter_search", {})
        self.best_params = {}
        self.best_score = None
        self.best_estimator = None
        
    def grid_search(self, model, param_grid: Dict[str, List], X, y, 
                   cv: int = 5, scoring: str = 'roc_auc', n_jobs: int = -1) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning."""
        try:
            # Create scoring
            if scoring == 'roc_auc':
                scorer = make_scorer(roc_auc_score, needs_proba=True)
            else:
                scorer = scoring
            
            # Initialize grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            # Get best parameters and score
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.best_estimator = grid_search.best_estimator_
            
            results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_estimator': self.best_estimator,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Grid search completed. Best params: {self.best_params}")
            logger.info(f"Best score: {self.best_score}")
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Grid search error: {str(e)}")
    
    def random_search(self, model, param_distributions: Dict[str, Any], X, y,
                     n_iter: int = 100, cv: int = 5, scoring: str = 'roc_auc',
                     n_jobs: int = -1) -> Dict[str, Any]:
        """Perform random search for hyperparameter tuning."""
        try:
            # Create scoring
            if scoring == 'roc_auc':
                scorer = make_scorer(roc_auc_score, needs_proba=True)
            else:
                scorer = scoring
            
            # Initialize random search
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs,
                verbose=1,
                random_state=42
            )
            
            # Fit random search
            random_search.fit(X, y)
            
            # Get best parameters and score
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            self.best_estimator = random_search.best_estimator_
            
            results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_estimator': self.best_estimator,
                'cv_results': random_search.cv_results_
            }
            
            logger.info(f"Random search completed. Best params: {self.best_params}")
            logger.info(f"Best score: {self.best_score}")
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Random search error: {str(e)}")
    
    def tune_model(self, model, model_name: str, X, y, method: str = 'grid') -> Dict[str, Any]:
        """Tune hyperparameters for a specific model."""
        try:
            param_grid = self.tuning_config.get(model_name, {}).get("param_grid", {})
            
            if not param_grid:
                logger.warning(f"No param grid found for {model_name}, skipping tuning")
                return {'best_params': {}, 'best_score': None}
            
            if method == 'grid':
                results = self.grid_search(model, param_grid, X, y)
            elif method == 'random':
                results = self.random_search(model, param_grid, X, y)
            else:
                raise ValueError(f"Unknown tuning method: {method}")
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Model tuning error: {str(e)}")