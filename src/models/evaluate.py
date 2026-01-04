import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, 
                           auc, confusion_matrix, classification_report)
from typing import Dict, Tuple
import json

class ModelEvaluator:
    """Model evaluation and visualization"""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model") -> Dict:
        """Evaluate model and return metrics"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / \
                             (metrics['precision'] + metrics['sensitivity']) \
                             if (metrics['precision'] + metrics['sensitivity']) > 0 else 0
        
        self.metrics[model_name] = metrics
        return metrics
    
    def plot_roc_curve(self, model, X_test, y_test, ax=None):
        """Plot ROC curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_precision_recall_curve(self, model, X_test, y_test, ax=None):
        """Plot Precision-Recall curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_confusion_matrix(self, model, X_test, y_test, ax=None):
        """Plot confusion matrix"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        
        return ax
    
    def plot_feature_importance(self, model, feature_names, top_n: int = 20, ax=None):
        """Plot feature importance for tree-based models"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            # Plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Feature Importances')
            ax.grid(True, alpha=0.3, axis='x')
        
        return ax
    
    def save_evaluation_report(self, model_name: str, metrics: Dict, 
                             filepath: str = 'monitoring/model_metrics.json'):
        """Save evaluation metrics to JSON"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Evaluation report saved to {filepath}")