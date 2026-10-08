import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..utils.logger import default_logger as logger
from ..utils.config import config_loader
from ..utils.helpers import load_artifact
from ..utils.exceptions import ModelPredictionError


class InferencePipeline:
    """Handle model inference and predictions."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")

        self.data_preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer()

        self.model = None
        self.preprocessor = None
        self.features = None
        self.threshold = 0.5

        self.load_artifacts()

    def load_artifacts(self):
        """Load the trained model and inference artifacts."""
        try:
            artifacts_dir = Path("artifacts")

            # Load the current trained model.
            model_files = sorted(
                (artifacts_dir / "trained_models").glob("*.pkl")
            )

            if not model_files:
                raise ModelPredictionError(
                    "No trained model found in artifacts/trained_models/"
                )

            model_path = model_files[-1]
            self.model = load_artifact(model_path)

            logger.info(
                f"Model loaded successfully: {model_path}"
            )

            # Load fitted preprocessor.
            preprocessor_path = artifacts_dir / "preprocessor.pkl"

            if not preprocessor_path.exists():
                raise ModelPredictionError(
                    "Preprocessor artifact not found"
                )

            self.preprocessor = load_artifact(preprocessor_path)
            self.data_preprocessor.preprocessor = self.preprocessor

            logger.info("Preprocessor loaded successfully")

            # Load feature list if available.
            features_path = (
                artifacts_dir / "feature_lists" / "features.json"
            )

            if features_path.exists():
                import json
                import pickle

                try:
                    with open(features_path, "r", encoding="utf-8") as f:
                        self.features = json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    with open(features_path, "rb") as f:
                        self.features = pickle.load(f)

                if isinstance(self.features, dict):
                    self.features = self.features.get(
                        "features",
                        self.features.get("feature_names", [])
                    )

                logger.info(
                    f"Feature list loaded: {len(self.features)} features"
                )

            # Load threshold.
            threshold_path = (
                artifacts_dir / "threshold_config" / "threshold.json"
            )

            if threshold_path.exists():
                threshold_data = load_artifact(threshold_path)

                if isinstance(threshold_data, dict):
                    self.threshold = float(
                        threshold_data.get("threshold", 0.5)
                    )

            logger.info(
                f"Threshold loaded: {self.threshold}"
            )

        except Exception as e:
            logger.error(
                f"Error loading artifacts: {str(e)}"
            )
            raise

    @staticmethod
    def _remove_transformer_prefixes(
        feature_names: List[str]
    ) -> List[str]:
        """Match feature names used by the trained model."""
        return [
            name.split("__", 1)[1]
            if "__" in name
            else name
            for name in feature_names
        ]

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new customer data."""
        try:
            if self.model is None:
                raise ModelPredictionError("Model not loaded")

            if df.empty:
                raise ModelPredictionError(
                    "Input dataframe is empty"
                )

            # Step 1: Clean data.
            df_clean = self.data_preprocessor.clean_data(df)

            # Step 2: Remove ID and target columns if present.
            columns_to_drop = [
                col
                for col in (
                    self.data_preprocessor.id_columns
                    + [self.data_preprocessor.target_column]
                )
                if col in df_clean.columns
            ]

            X = df_clean.drop(columns=columns_to_drop)

            # Step 3: Feature engineering.
            X_engineered = self.feature_engineer.create_features(X)

            # Step 4: Apply the fitted training preprocessor.
            if self.preprocessor is None:
                raise ModelPredictionError(
                    "Preprocessor not loaded"
                )

            X_processed = self.preprocessor.transform(
                X_engineered
            )

            feature_names = (
                self.preprocessor.get_feature_names_out()
            )

            feature_names = self._remove_transformer_prefixes(
                feature_names.tolist()
            )

            X_processed = pd.DataFrame(
                X_processed,
                columns=feature_names,
                index=X_engineered.index
            )

            # Step 5: Ensure the model receives the exact
            # feature set used during training.
            if self.features:
                missing_features = [
                    feature
                    for feature in self.features
                    if feature not in X_processed.columns
                ]

                if missing_features:
                    raise ModelPredictionError(
                        "Missing inference features: "
                        f"{missing_features}"
                    )

                X_final = X_processed[self.features]
            else:
                X_final = X_processed

            logger.info(
                f"Inference feature shape: {X_final.shape}"
            )

            # Step 6: Model prediction.
            y_prob = self.model.predict_proba(
                X_final
            )[:, 1]

            # Step 7: Apply configured threshold.
            y_pred = (
                y_prob >= self.threshold
            ).astype(int)

            # Step 8: Format results.
            results = []

            for idx in range(len(df)):
                customer_id = (
                    df.iloc[idx]["customerID"]
                    if "customerID" in df.columns
                    else idx
                )

                probability = float(y_prob[idx])

                results.append({
                    "customer_id": customer_id,
                    "churn_prediction": int(y_pred[idx]),
                    "churn_probability": probability,
                    "risk_level": self._get_risk_level(
                        probability
                    )
                })

            logger.info(
                f"Predictions completed for {len(df)} customers"
            )

            return {
                "predictions": results,
                "predicted_labels": y_pred.tolist(),
                "probabilities": y_prob.tolist()
            }

        except Exception as e:
            raise ModelPredictionError(
                f"Prediction error: {str(e)}"
            )

    def _get_risk_level(self, probability: float) -> str:
        """Get risk level based on churn probability."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        return "High"

    def batch_predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Make predictions in batches."""
        results = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            batch_results = self.predict(batch)

            results.extend(
                batch_results["predictions"]
            )

        return results
