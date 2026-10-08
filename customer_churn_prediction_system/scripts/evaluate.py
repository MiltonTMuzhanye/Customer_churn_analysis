import sys
import os
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.customer_churn.data.preprocessing import DataPreprocessor
from src.customer_churn.features.engineering import FeatureEngineer
from src.customer_churn.evaluation.metrics import MetricsCalculator
from src.customer_churn.evaluation.threshold_analysis import ThresholdAnalyzer
from src.customer_churn.evaluation.explainability import ModelExplainer
from src.customer_churn.evaluation.validation import ModelValidator
from src.customer_churn.utils.helpers import load_artifact
from src.customer_churn.utils.logger import default_logger as logger


def main():
    """Evaluate a trained customer churn model."""

    parser = argparse.ArgumentParser(
        description="Evaluate customer churn model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to raw dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/",
        help="Directory to save evaluation reports"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting model evaluation")

        output_dir = Path(args.output_dir)
        metrics_dir = output_dir / "metrics"
        figures_dir = output_dir / "figures"

        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        model = load_artifact(args.model_path)
        df = pd.read_csv(args.test_data)

        logger.info(
            f"Loaded evaluation dataset: {df.shape}"
        )

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)

        y = preprocessor.prepare_target(df_clean)

        X = df_clean.drop(
            columns=[
                preprocessor.target_column,
                *preprocessor.id_columns
            ]
        )

        feature_engineer = FeatureEngineer()
        X_engineered = feature_engineer.create_features(X)

        logger.info(
            f"Engineered evaluation features: {X_engineered.shape}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        logger.info(
            f"Evaluation test shape: {X_test.shape}"
        )

        saved_preprocessor = load_artifact(
            "artifacts/preprocessor.pkl"
        )

        X_test_processed = saved_preprocessor.transform(X_test)

        feature_names = saved_preprocessor.get_feature_names_out()

        # Match the feature names used when the model was trained.
        feature_names = [
            name.split("__", 1)[1]
            if "__" in name
            else name
            for name in feature_names
        ]

        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index
        )

        logger.info(
            f"Processed evaluation features: "
            f"{X_test_processed.shape}"
        )

        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        metrics_calc = MetricsCalculator()

        metrics = metrics_calc.calculate_all_metrics(
            y_test,
            y_pred,
            y_prob
        )

        logger.info(
            f"Evaluation metrics: {metrics}"
        )

        metrics_df = pd.DataFrame([metrics])

        metrics_df.to_csv(
            metrics_dir / "performance_metrics.csv",
            index=False
        )

        threshold_analyzer = ThresholdAnalyzer()

        threshold_analyzer.analyze_thresholds(
            y_test,
            y_prob
        )

        threshold_analyzer.plot_threshold_curves(
            str(figures_dir / "threshold_analysis.png")
        )

        explainer = ModelExplainer(
            model,
            X_test_processed
        )

        shap_values = explainer.get_shap_values(
            X_test_processed
        )

        explainer.plot_shap_summary(
            shap_values,
            X_test_processed,
            str(figures_dir / "shap_summary.png")
        )

        explainer.plot_shap_bar(
            shap_values,
            str(figures_dir / "shap_bar.png")
        )

        validator = ModelValidator()

        cv_results = validator.cross_validation(
            model,
            X_test_processed,
            y_test
        )

        logger.info(
            f"Cross-validation results: "
            f"{cv_results['metrics']}"
        )

        logger.info(
            "Evaluation completed successfully"
        )

    except Exception as e:
        logger.error(
            f"Evaluation failed: {str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()