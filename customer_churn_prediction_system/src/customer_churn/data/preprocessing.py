import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List
from ..utils.logger import default_logger as logger
from ..utils.exceptions import FeatureEngineeringError
from ..utils.config import config_loader

class DataPreprocessor:
    """Handle data preprocessing and transformation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("config")
        self.feature_config = self.config.get("features", {})
        self.target_column = self.config.get("data", {}).get("target_column", "Churn")
        self.id_columns = self.feature_config.get("id_columns", ["customerID"])
        
        self.numerical_features = self.feature_config.get("numerical_features", [])
        self.categorical_features = self.feature_config.get("categorical_features", [])
        
        self.preprocessor = None
        self.scalers = {}
        self.encoders = {}
        self.target_encoder = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data."""
        try:
            df = df.copy()

            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"],
                    errors="coerce"
                )

                median_total_charges = df["TotalCharges"].median()

                df["TotalCharges"] = df["TotalCharges"].fillna(
                    median_total_charges
                )

            for col in df.columns:

                if df[col].isnull().any():

                    if pd.api.types.is_numeric_dtype(df[col]):
                        median_value = df[col].median()
                        df[col] = df[col].fillna(median_value)

                    else:
                        mode = df[col].mode()

                        if not mode.empty:
                            df[col] = df[col].fillna(mode.iloc[0])
                        else:
                            df[col] = df[col].fillna("Unknown")

            logger.info(f"Data cleaned. Shape: {df.shape}")

            return df

        except Exception as e:
            raise FeatureEngineeringError(
                f"Data cleaning error: {str(e)}"
            )
    
    def create_preprocessor(self, df: pd.DataFrame, fit: bool = True) -> Pipeline:
        """Create a preprocessing pipeline using all available feature columns."""
        try:
            excluded_columns = set(
                self.id_columns + [self.target_column]
            )

            # Detect features from the actual dataframe so engineered
            # features are included automatically.
            self.numerical_features = [
                col
                for col in df.select_dtypes(include=[np.number]).columns
                if col not in excluded_columns
            ]

            self.categorical_features = [
                col
                for col in df.select_dtypes(
                    include=["object", "category"]
                ).columns
                if col not in excluded_columns
            ]

            numerical_transformer = StandardScaler()

            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numerical_transformer,
                        self.numerical_features
                    ),
                    (
                        "cat",
                        categorical_transformer,
                        self.categorical_features
                    )
                ],
                remainder="drop"
            )

            if fit:
                X = df[
                    self.numerical_features +
                    self.categorical_features
                ]

                preprocessor.fit(X)
                self.preprocessor = preprocessor

            logger.info(
                "Preprocessor created successfully. "
                f"Numerical features: {len(self.numerical_features)}, "
                f"Categorical features: {len(self.categorical_features)}"
            )

            return preprocessor

        except Exception as e:
            raise FeatureEngineeringError(
                f"Preprocessor creation error: {str(e)}"
            )

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the preprocessor."""
        try:
            if self.preprocessor is None:
                self.create_preprocessor(df, fit=True)
            
            # Separate features
            X = df[self.numerical_features + self.categorical_features]
            
            # Transform
            X_transformed = self.preprocessor.transform(X)
            
            # Get feature names
            feature_names = self.get_feature_names()
            
            # Create DataFrame
            X_processed = pd.DataFrame(X_transformed, columns=feature_names)
            
            # Add IDs if present
            for id_col in self.id_columns:
                if id_col in df.columns:
                    X_processed[id_col] = df[id_col].values
            
            logger.info(f"Data transformed. Shape: {X_processed.shape}")
            return X_processed
            
        except Exception as e:
            raise FeatureEngineeringError(f"Data transformation error: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        try:
            if self.preprocessor is None:
                return []
            
            feature_names = []
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(columns))
                    else:
                        # Fallback for older sklearn versions
                        import re
                        for col in columns:
                            unique_values = transformer.categories_[transformer.transformers_.index((name, transformer, columns))]
                            feature_names.extend([f"{col}_{val}" for val in unique_values])
            
            return feature_names
            
        except Exception as e:
            raise FeatureEngineeringError(f"Error getting feature names: {str(e)}")
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable."""
        try:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column {self.target_column} not found")
            
            y = df[self.target_column].copy()
            
            # Encode target if needed
            if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
                if self.target_encoder is None:
                    self.target_encoder = LabelEncoder()
                    y_encoded = self.target_encoder.fit_transform(y)
                else:
                    y_encoded = self.target_encoder.transform(y)

                return pd.Series(y_encoded, index=y.index, name=self.target_column)

            else:
                return y
            
        except Exception as e:
            raise FeatureEngineeringError(f"Target preparation error: {str(e)}")
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform target variable."""
        try:
            if self.target_encoder is not None:
                return self.target_encoder.inverse_transform(y_encoded)
            return y_encoded
        except Exception as e:
            raise FeatureEngineeringError(f"Target inverse transform error: {str(e)}")