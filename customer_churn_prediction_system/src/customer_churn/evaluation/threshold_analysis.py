import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from ..utils.logger import default_logger as logger
from ..utils.exceptions import ModelTrainingError

class ThresholdAnalyzer:
    """Analyze and optimize decision threshold."""
    
    def __init__(self):
        self.threshold_results = None
        self.optimal_thresholds = {}
        
    def analyze_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           thresholds: List[float] = None) -> pd.DataFrame:
        """Analyze model performance across different thresholds."""
        try:
            if thresholds is None:
                thresholds = np.linspace(0, 1, 100)
            
            results = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                
                metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                }
                results.append(metrics)
            
            self.threshold_results = pd.DataFrame(results)
            
            # Find optimal thresholds for each metric
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                optimal_idx = self.threshold_results[metric].idxmax()
                self.optimal_thresholds[metric] = self.threshold_results.loc[optimal_idx, 'threshold']
            
            logger.info(f"Optimal thresholds: {self.optimal_thresholds}")
            return self.threshold_results
            
        except Exception as e:
            raise ModelTrainingError(f"Threshold analysis error: {str(e)}")
    
    def get_optimal_threshold(self, metric: str = 'f1') -> float:
        """Get optimal threshold for a specific metric."""
        return self.optimal_thresholds.get(metric, 0.5)
    
    def plot_threshold_curves(self, save_path: str = None):
        """Plot metrics vs threshold."""
        if self.threshold_results is None:
            raise ValueError("Run analyze_thresholds first")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            ax.plot(self.threshold_results['threshold'], self.threshold_results[metric])
            ax.axvline(self.optimal_thresholds[metric], color='red', linestyle='--', 
                      label=f'Optimal ({self.optimal_thresholds[metric]:.2f})')
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Threshold')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()