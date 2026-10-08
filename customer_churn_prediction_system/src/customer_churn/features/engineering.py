import pandas as pd
import numpy as np
from typing import List

from ..utils.logger import default_logger as logger
from ..utils.exceptions import FeatureEngineeringError


class FeatureEngineer:
    """Create deterministic, reproducible customer features."""

    def __init__(self):
        self.created_features = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features without relying on the current batch's values."""
        try:
            df = df.copy()
            self.created_features = []

            # Ensure numerical inputs are actually numeric.
            if "tenure" in df.columns:
                df["tenure"] = pd.to_numeric(
                    df["tenure"], errors="coerce"
                )

            if "MonthlyCharges" in df.columns:
                df["MonthlyCharges"] = pd.to_numeric(
                    df["MonthlyCharges"], errors="coerce"
                )

            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"], errors="coerce"
                )

            if "tenure" in df.columns:
                df["tenure_category"] = pd.cut(
                    df["tenure"],
                    bins=[-1, 0, 12, 24, 48, 72],
                    labels=[
                        "New",
                        "Short-term",
                        "Medium-term",
                        "Long-term",
                        "Very-long-term",
                    ],
                    include_lowest=True,
                )
                self.created_features.append("tenure_category")

            if all(
                col in df.columns
                for col in ["TotalCharges", "tenure", "MonthlyCharges"]
            ):
                valid_tenure = df["tenure"].where(df["tenure"] > 0)
                df["avg_monthly_charges"] = (
                    df["TotalCharges"].div(valid_tenure)
                    .fillna(df["MonthlyCharges"])
                )
                self.created_features.append("avg_monthly_charges")

            service_count = pd.Series(0, index=df.index, dtype="int64")
            service_found = False

            yes_only_services = [
                "PhoneService",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]

            for col in yes_only_services:
                if col in df.columns:
                    service_count += df[col].eq("Yes").astype("int64")
                    service_found = True

            if "InternetService" in df.columns:
                service_count += (
                    df["InternetService"]
                    .isin(["DSL", "Fiber optic"])
                    .astype("int64")
                )
                service_found = True

            if service_found:
                df["service_count"] = service_count
                self.created_features.append("service_count")

            if all(
                col in df.columns for col in ["tenure", "MonthlyCharges"]
            ):
                df["tenure_monthly_charges"] = (
                    df["tenure"] * df["MonthlyCharges"]
                )
                self.created_features.append("tenure_monthly_charges")

            if "Contract" in df.columns:
                df["is_month_to_month"] = (
                    df["Contract"] == "Month-to-month"
                ).astype(int)
                df["is_one_year"] = (
                    df["Contract"] == "One year"
                ).astype(int)
                df["is_two_year"] = (
                    df["Contract"] == "Two year"
                ).astype(int)

                self.created_features.extend([
                    "is_month_to_month",
                    "is_one_year",
                    "is_two_year",
                ])

            if "PaymentMethod" in df.columns:
                df["is_electronic_check"] = (
                    df["PaymentMethod"] == "Electronic check"
                ).astype(int)
                self.created_features.append("is_electronic_check")

            if all(
                col in df.columns for col in ["MonthlyCharges", "tenure"]
            ):
                monthly_norm = (
                    df["MonthlyCharges"].clip(lower=0, upper=120) / 120
                ).fillna(0)

                tenure_norm = (
                    df["tenure"].clip(lower=0, upper=72) / 72
                ).fillna(0)

                df["customer_value_score"] = (
                    monthly_norm * 0.6 + tenure_norm * 0.4
                )
                self.created_features.append("customer_value_score")

            if all(
                col in df.columns
                for col in ["tenure", "Contract", "PaperlessBilling"]
            ):
                risk_factors = pd.DataFrame(index=df.index)
                risk_factors["tenure_risk"] = (
                    df["tenure"] < 12
                ).astype(int)
                risk_factors["contract_risk"] = (
                    df["Contract"] == "Month-to-month"
                ).astype(int)
                risk_factors["paperless_risk"] = (
                    df["PaperlessBilling"] == "Yes"
                ).astype(int)

                df["risk_score"] = risk_factors.sum(axis=1) / 3
                self.created_features.append("risk_score")

            logger.info(
                "Created %d engineered features",
                len(self.created_features),
            )
            return df

        except Exception as e:
            raise FeatureEngineeringError(
                f"Feature creation error: {str(e)}"
            ) from e

    def get_feature_importance_ranking(
        self, model, feature_names: List[str]
    ) -> pd.DataFrame:
        """Get feature importance rankings."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                raise ValueError(
                    "Model does not have feature_importances_ or coef_"
                )

            return pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)

        except Exception as e:
            raise FeatureEngineeringError(
                f"Feature importance calculation error: {str(e)}"
            ) from e