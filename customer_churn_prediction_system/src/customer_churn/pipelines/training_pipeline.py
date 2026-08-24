import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import mlflow
from ..data.ingestion import DataIngestor
from ..data.validation import DataValidator
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..features.selection import FeatureSelector
from ..training.trainer import ModelTrainer
from ..training.hyperparameter_tuning import HyperparameterTuner
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.explainability import ModelExplainer
from ..evaluation.threshold_analysis import ThresholdAnalyzer
from ..evaluation.validation import ModelValidator
from ..utils.logger import default_logger as logger
from ..utils.config import config_loader
from ..utils.helpers import save_artifact, load_artifact
from ..utils.exceptions import ModelTrainingError

class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.data_ingestor = DataIngestor(config)
        self.data_validator = DataValidator()
        self.data_preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer(config)
        self.hyperparameter_tuner = HyperparameterTuner()
        self.metrics_calculator = MetricsCalculator()
        self.threshold_analyzer = ThresholdAnalyzer()
        self.model_validator = ModelValidator()
        
        self.df_processed = None
        self.X_processed = None
        self.y_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.probabilities = None
        
    def run(self, model_class, model_name: str, **kwargs) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            logger.info(f"Starting training pipeline for {model_name}")
            
            # Step 1: Load data
            df_raw = self.data_ingestor.load_data()
            
            # Step 2: Validate data
            valid, validation_summary = self.data_validator.validate_all(df_raw)
            if not valid:
                logger.warning("Data validation failed, continuing with warnings")
            
            # Step 3: Preprocess data
            df_clean = self.data_preprocessor.clean_data(df_raw)
            X = df_clean.drop(columns=self.data_preprocessor.id_columns + [self.data_preprocessor.target_column])
            y = self.data_preprocessor.prepare_target(df_clean)
            
            # Step 4: Feature engineering
            X_engineered = self.feature_engineer.create_features(X)
            
            # Step 5: Preprocess features
            self.data_preprocessor.create_preprocessor(X_engineered, fit=True)
            X_processed = self.data_preprocessor.transform_data(X_engineered)
            
            # Step 6: Feature selection (optional)
            if kwargs.get('feature_selection', False):
                X_selected = self.feature_selector.select_features(X_processed, y)
                X_final = X_selected
            else:
                X_final = X_processed
            
            self.X_processed = X_final
            self.y_processed = y
            
            # Step 7: Split data
            X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(X_final, y)
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            # Step 8: Hyperparameter tuning (if enabled)
            if self.config.get('model', {}).get('hyperparameter_tuning', False):
                tuner = HyperparameterTuner()
                model_temp = model_class({}).create_model()
                tuning_results = tuner.tune_model(
                    model_temp, model_name, X_train, y_train, 
                    method=kwargs.get('tuning_method', 'grid')
                )
                best_params = tuning_results.get('best_params', {})
            else:
                best_params = {}
            
            # Step 9: Train model
            model = self.model_trainer.train_model(
                model_class, model_name, X_train, y_train, **best_params
            )
            self.model = model
            
            # Step 10: Evaluate model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            self.predictions = y_pred
            self.probabilities = y_prob
            
            metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred, y_prob)
            
            # Step 11: Threshold analysis
            threshold_results = self.threshold_analyzer.analyze_thresholds(y_test, y_prob)
            optimal_threshold = self.threshold_analyzer.get_optimal_threshold('f1')
            
            # Step 12: Cross-validation
            cv_results = self.model_validator.cross_validation(model, X_train, y_train)
            
            # Step 13: Model explainability
            explainer = ModelExplainer(model, X_train)
            shap_values = explainer.get_shap_values(X_test)
            feature_importance = explainer.calculate_permutation_importance(X_test, y_test)
            
            # Step 14: Save model and artifacts
            self.model_trainer.save_model(
                metadata={
                    'metrics': metrics,
                    'cv_results': cv_results,
                    'optimal_threshold': optimal_threshold,
                    'feature_importance': feature_importance.to_dict() if feature_importance is not None else {},
                    'model_name': model_name
                }
            )
            
            # Step 15: Save artifacts
            self.save_artifacts()
            
            # Step 16: Log with MLflow
            with mlflow.start_run(run_name=f"{model_name}_pipeline"):
                mlflow.log_params(self.model_trainer.model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.log_artifact("artifacts/")
            
            results = {
                'model': model,
                'metrics': metrics,
                'cv_results': cv_results,
                'threshold': optimal_threshold,
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            logger.info(f"Training pipeline completed for {model_name}")
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Training pipeline error: {str(e)}")
    
    def save_artifacts(self):
        """Save pipeline artifacts."""
        try:
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessor
            save_artifact(self.data_preprocessor.preprocessor, artifacts_dir / "preprocessor.pkl")
            
            # Save feature list
            feature_list = list(self.X_processed.columns)
            save_artifact(feature_list, artifacts_dir / "feature_lists/features.json")
            
            # Save threshold
            save_artifact(
                {'threshold': self.threshold_analyzer.get_optimal_threshold()},
                artifacts_dir / "threshold_config/threshold.json"
            )
            
            logger.info("Artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")