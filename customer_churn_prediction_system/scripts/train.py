import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.customer_churn.pipelines.training_pipeline import TrainingPipeline
from src.customer_churn.models.logistic_regression import LogisticRegressionModel
from src.customer_churn.models.random_forest import RandomForestModel
from src.customer_churn.models.xgboost_model import XGBoostModel
from src.customer_churn.models.lightgbm_model import LightGBMModel
from src.customer_churn.models.catboost_model import CatBoostModel
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

MODEL_MAP = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel
}

def main():
    """Train a model using the training pipeline."""
    parser = argparse.ArgumentParser(description='Train customer churn model')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost'],
                       help='Model to train')
    parser.add_argument('--tuning', action='store_true', 
                       help='Enable hyperparameter tuning')
    parser.add_argument('--tuning-method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Hyperparameter tuning method')
    parser.add_argument('--feature-selection', action='store_true',
                       help='Enable feature selection')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting training for {args.model}")
        
        # Load config
        config = config_loader.get_config("config")
        model_config = config.get("model", {})
        
        # Update config with arguments
        if args.tuning:
            model_config['hyperparameter_tuning'] = True
        
        # Get model class
        model_class = MODEL_MAP.get(args.model)
        if model_class is None:
            raise ValueError(f"Unknown model: {args.model}")
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Run pipeline
        results = pipeline.run(
            model_class, 
            args.model,
            feature_selection=args.feature_selection,
            tuning_method=args.tuning_method
        )
        
        logger.info(f"Training completed for {args.model}")
        logger.info(f"Metrics: {results['metrics']}")
        logger.info(f"CV Mean Score: {results['cv_results']['cv_mean']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()