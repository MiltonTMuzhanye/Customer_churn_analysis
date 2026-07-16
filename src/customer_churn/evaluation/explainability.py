# src/customer_churn/evaluation/explainability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class ModelExplainer:
    """Explain model predictions and behavior."""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_importance = None
        
    def create_shap_explainer(self, method: str = 'tree'):
        """Create SHAP explainer."""
        try:
            if method == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif method == 'kernel':
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, self.X_train
                )
            elif method == 'linear':
                self.shap_explainer = shap.LinearExplainer(
                    self.model, self.X_train
                )
            else:
                raise ValueError(f"Unknown SHAP method: {method}")
            
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"SHAP explainer creation error: {str(e)}")
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for predictions."""
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        try:
            shap_values = self.shap_explainer.shap_values(X)
            return shap_values
        except Exception as e:
            raise ModelTrainingError(f"SHAP value calculation error: {str(e)}")
    
    def create_lime_explainer(self):
        """Create LIME explainer."""
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification',
                training_labels=None
            )
            logger.info("LIME explainer created successfully")
        except Exception as e:
            raise ModelTrainingError(f"LIME explainer creation error: {str(e)}")
    
    def get_lime_explanation(self, instance: pd.Series, num_features: int = 10) -> dict:
        """Get LIME explanation for a single instance."""
        if self.lime_explainer is None:
            self.create_lime_explainer()
        
        try:
            explanation = self.lime_explainer.explain_instance(
                instance.values,
                self.model.predict_proba,
                num_features=num_features
            )
            return explanation
        except Exception as e:
            raise ModelTrainingError(f"LIME explanation error: {str(e)}")
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame = None, 
                         save_path: str = None):
        """Plot SHAP summary plot."""
        X = X or self.X_train
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_bar(self, shap_values: np.ndarray, save_path: str = None):
        """Plot SHAP bar plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_train, plot_type='bar', 
                         feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series, 
                                        n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation feature importance."""
        try:
            result = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            return importance_df
            
        except Exception as e:
            raise ModelTrainingError(f"Permutation importance calculation error: {str(e)}")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame = None, 
                               save_path: str = None):
        """Plot feature importance."""
        importance_df = importance_df or self.feature_importance
        if importance_df is None:
            raise ValueError("No feature importance data available")
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()