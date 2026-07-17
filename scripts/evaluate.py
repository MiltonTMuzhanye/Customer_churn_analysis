# scripts/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.customer_churn.evaluation.metrics import MetricsCalculator
from src.customer_churn.evaluation.threshold_analysis import ThresholdAnalyzer
from src.customer_churn.evaluation.explainability import ModelExplainer
from src.customer_churn.evaluation.validation import ModelValidator
from src.customer_churn.utils.helpers import load_artifact
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

def main():
    """Evaluate model performance."""
    parser = argparse.ArgumentParser(description='Evaluate customer churn model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='reports/',
                       help='Directory to save evaluation reports')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting model evaluation")
        
        # Load model and test data
        model = load_artifact(args.model_path)
        test_df = pd.read_csv(args.test_data)
        
        # Prepare data
        X_test = test_df.drop(columns=['Churn', 'customerID'])
        y_test = test_df['Churn'].map({'Yes': 1, 'No': 0}).values
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, y_prob)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{args.output_dir}/metrics/performance_metrics.csv", index=False)
        
        # Threshold analysis
        threshold_analyzer = ThresholdAnalyzer()
        threshold_analyzer.analyze_thresholds(y_test, y_prob)
        threshold_analyzer.plot_threshold_curves(
            f"{args.output_dir}/figures/threshold_analysis.png"
        )
        
        # Model explainability
        explainer = ModelExplainer(model, X_test)
        shap_values = explainer.get_shap_values(X_test)
        explainer.plot_shap_summary(shap_values, X_test, 
                                   f"{args.output_dir}/figures/shap_summary.png")
        explainer.plot_shap_bar(shap_values, 
                               f"{args.output_dir}/figures/shap_bar.png")
        
        # Cross-validation
        validator = ModelValidator()
        cv_results = validator.cross_validation(model, X_test, y_test)
        logger.info(f"Cross-validation results: {cv_results['metrics']}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()