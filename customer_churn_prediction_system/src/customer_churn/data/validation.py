import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ..utils.logger import default_logger as logger
from ..utils.exceptions import DataValidationError
from ..utils.config import config_loader

class DataValidator:
    """Validate data quality and schema."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or config_loader.get_config("data")
        self.validation_config = self.config.get("data_validation", {})
        self.schema = self.validation_config.get("schema", {})
        self.quality_config = self.validation_config.get("quality_checks", {})
        
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate data against schema."""
        try:
            errors = []
            
            for column, rules in self.schema.items():
                if column not in df.columns:
                    if rules.get('required', False):
                        errors.append(f"Required column missing: {column}")
                    continue

                series = df[column]
                
                # Check type
                expected_type = rules.get('type')
                if expected_type == "string":
                    if not (
                        pd.api.types.is_object_dtype(series)
                        or pd.api.types.is_string_dtype(series)
                    ):
                        errors.append(
                            f"Column {column} should be string type, "
                            f"got {series.dtype}"
                        )

                elif expected_type == "integer":

                    if not pd.api.types.is_integer_dtype(series):
                        numeric_series = pd.to_numeric(
                            series,
                            errors="coerce"
                        )

                        invalid_numeric = numeric_series.isna().sum()

                        if invalid_numeric > 0:
                            errors.append(
                                f"Column {column} contains "
                                f"{invalid_numeric} non-numeric values"
                            )
                        elif not (
                            numeric_series.dropna() % 1 == 0
                        ).all():
                            errors.append(
                                f"Column {column} contains "
                                f"non-integer numeric values"
                            )

                elif expected_type == "float":

                    numeric_series = pd.to_numeric(
                        series,
                        errors="coerce"
                    )

                    non_numeric = (
                        numeric_series.isna()
                        & series.notna()
                        & (series.astype(str).str.strip() != "")
                    )

                    if non_numeric.any():
                        errors.append(
                            f"Column {column} contains "
                            f"non-numeric values"
                        )
                
                # Check allowed values
                allowed_values = rules.get("allowed_values")
                if allowed_values:
                    cleaned_values = series.dropna()

                    invalid_values = (
                        set(cleaned_values.unique())
                        - set(allowed_values)
                    )

                    # For raw numeric columns, compare numerically
                    if expected_type in ["integer", "float"]:
                        numeric_allowed = set(
                            pd.to_numeric(
                                list(allowed_values),
                                errors="coerce"
                            )
                        )

                        numeric_values = pd.to_numeric(
                            cleaned_values,
                            errors="coerce"
                        ).dropna()

                        invalid_numeric = (
                            set(numeric_values)
                            - numeric_allowed
                        )

                        invalid_values = invalid_numeric

                    if invalid_values:
                        errors.append(
                            f"Column {column} contains invalid "
                            f"values: {invalid_values}"
                        )

                
                # Check min
                min_value = rules.get("min_value")

                if min_value is not None:
                    numeric_series = pd.to_numeric(
                        series,
                        errors="coerce"
                    )

                    if (
                        numeric_series.dropna() < min_value
                    ).any():
                        errors.append(
                            f"Column {column} has values below "
                            f"{min_value}"
                        )

                # Check max
                max_value = rules.get("max_value")

                if max_value is not None:
                    numeric_series = pd.to_numeric(
                        series,
                        errors="coerce"
                    )

                    if (
                        numeric_series.dropna() > max_value
                    ).any():
                        errors.append(
                            f"Column {column} has values above "
                            f"{max_value}"
                        )


            
            if errors:
                raise DataValidationError(f"Schema validation failed: {errors}")
            
            logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            raise DataValidationError(f"Schema validation error: {str(e)}")
    
    def check_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality checks."""
        try:
            results = {
                'passed': True,
                'checks': {}
            }
            
            # Check for missing values
            missing_threshold = self.quality_config.get('missing_threshold', 0.0)
            missing_columns = []
            for col in df.columns:
                missing_pct = df[col].isnull().mean()

                if missing_pct > missing_threshold:
                    missing_columns.append({
                        'column': col,
                        'missing_pct': missing_pct
                    })
            
            results['checks']['missing_values'] = {
                'passed': len(missing_columns) == 0,
                'details': missing_columns
            }
            if missing_columns:
                results['passed'] = False
            
            # Check for duplicates
            duplicate_threshold = self.quality_config.get('duplicate_threshold', 0.0)
            duplicates = df.duplicated().sum()

            duplicate_pct = (
                duplicates / len(df)
                if len(df) > 0
                else 0
            )
            
            results['checks']['duplicates'] = {
                'passed': duplicate_pct <= duplicate_threshold,
                'count': duplicates,
                'percentage': duplicate_pct
            }
            if duplicate_pct > duplicate_threshold:
                results['passed'] = False
            
            # Check for outliers in numerical columns
            outlier_method = self.quality_config.get('outlier_method', 'iqr')
            outlier_threshold = self.quality_config.get('outlier_threshold', 1.5)
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numerical_cols:
                if outlier_method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_counts[col] = len(outliers)
                elif outlier_method == 'zscore':
                    std = df[col].std()

                    if std == 0:
                        outlier_counts[col] = 0
                    else:
                        z_scores = np.abs((df[col] - df[col].mean()) / std)
                        outlier_counts[col] = (z_scores > outlier_threshold).sum()
            
            results['checks']['outliers'] = {
                'passed': True,
                'details': outlier_counts
            }
            
            logger.info(f"Quality check results: {results}")
            return results
            
        except Exception as e:
            raise DataValidationError(f"Quality check error: {str(e)}")
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Run all validation checks."""
        try:
            schema_passed = self.validate_schema(df)
            quality_results = self.check_quality(df)
            
            all_passed = schema_passed and quality_results['passed']
            validation_summary = {
                'schema_passed': schema_passed,
                'quality_results': quality_results,
                'all_passed': all_passed
            }
            
            if not all_passed:
                logger.warning(f"Validation failed: {validation_summary}")
            else:
                logger.info("All validation checks passed")
            
            return all_passed, validation_summary
            
        except Exception as e:
            raise DataValidationError(f"Validation error: {str(e)}")