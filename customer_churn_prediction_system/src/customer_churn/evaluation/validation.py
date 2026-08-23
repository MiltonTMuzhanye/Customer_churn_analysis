import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict, Any, List, Tuple
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class ModelValidator:
    """Validate model performance."""
    
    def __init__(self):
        self.validation_results = {}
        
    def cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                        cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            # Map scoring metric
            scoring_map = {
                'roc_auc': 'roc_auc',
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1'
            }
            
            # Get predictions
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=kf, method='predict')
            
            # Get probabilities if available
            try:
                y_prob = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:, 1]
            except:
                y_prob = None
            
            # Calculate metrics
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                       f1_score, roc_auc_score)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred)
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y, y_prob)
            
            # Get cross-validation scores
            scorer = make_scorer(roc_auc_score, needs_proba=True) if scoring == 'roc_auc' else scoring
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            
            results = {
                'metrics': metrics,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_prob.tolist() if y_prob is not None else None
            }
            
            self.validation_results = results
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Cross-validation error: {str(e)}")
    
    def stability_test(self, model, X: pd.DataFrame, y: pd.Series, 
                      n_iterations: int = 10, test_size: float = 0.2) -> Dict[str, Any]:
        """Test model stability across multiple random splits."""
        try:
            results = []
            
            for i in range(n_iterations):
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=i, stratify=y
                )
                
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                y_pred = model_copy.predict(X_test)
                y_prob = model_copy.predict_proba(X_test)[:, 1]
                
                results.append({
                    'iteration': i,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_prob)
                })
            
            results_df = pd.DataFrame(results)
            
            summary = {
                'metrics': results_df.mean().to_dict(),
                'std': results_df.std().to_dict(),
                'results': results_df.to_dict('records')
            }
            
            logger.info(f"Stability test completed: {summary['metrics']}")
            return summary
            
        except Exception as e:
            raise ModelTrainingError(f"Stability test error: {str(e)}")