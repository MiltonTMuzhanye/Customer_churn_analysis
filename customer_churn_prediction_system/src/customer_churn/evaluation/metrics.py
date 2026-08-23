import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, log_loss, confusion_matrix,
                           classification_report, precision_recall_curve,
                           roc_curve, average_precision_score)
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class MetricsCalculator:
    """Calculate and manage evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics = {}
        self.confusion_matrix = None
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'f1': f1_score(y_true, y_pred, average='binary')
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['log_loss'] = log_loss(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            raise ModelTrainingError(f"Metrics calculation error: {str(e)}")
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix = cm
        return cm
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report."""
        report = classification_report(y_true, y_pred)
        return report
    
    def get_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                             metric: str = 'f1') -> float:
        """Find optimal threshold for a given metric."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        if metric == 'f1':
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'accuracy':
            # Need to calculate accuracy for each threshold
            accuracies = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                accuracies.append(accuracy_score(y_true, y_pred))
            optimal_idx = np.argmax(accuracies)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        return optimal_threshold
    
    def plot_confusion_matrix(self, cm: np.ndarray = None, save_path: str = None):
        """Plot confusion matrix."""
        cm = cm or self.confusion_matrix
        if cm is None:
            raise ValueError("Confusion matrix not available")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()