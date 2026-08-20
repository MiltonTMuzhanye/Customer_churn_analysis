import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Optional
from ..utils.logger import default_logger as logger
from ..utils.exceptions import FeatureEngineeringError

class FeatureSelector:
    """Handle feature selection."""
    
    def __init__(self, method: str = 'mutual_info', k: int = 20):
        self.method = method
        self.k = k
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select top k features using specified method."""
        try:
            # Remove non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]
            
            # Remove columns with all same values
            X_numeric = X_numeric.loc[:, X_numeric.nunique() > 1]
            
            if self.method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=min(self.k, X_numeric.shape[1]))
                selector.fit(X_numeric, y)
                scores = selector.scores_
                
            elif self.method == 'f_classif':
                selector = SelectKBest(f_classif, k=min(self.k, X_numeric.shape[1]))
                selector.fit(X_numeric, y)
                scores = selector.scores_
                
            elif self.method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator, n_features_to_select=min(self.k, X_numeric.shape[1]))
                selector.fit(X_numeric, y)
                scores = selector.ranking_
                
            else:
                raise ValueError(f"Unknown feature selection method: {self.method}")
            
            # Get feature scores
            self.feature_scores = pd.DataFrame({
                'feature': X_numeric.columns,
                'score': scores
            }).sort_values('score', ascending=False)
            
            # Select top features
            self.selected_features = self.feature_scores['feature'].head(self.k).tolist()
            
            logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
            return X[self.selected_features]
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature selection error: {str(e)}")
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature scores."""
        return self.feature_scores
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features