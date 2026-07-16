# src/customer_churn/data/validation.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
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
                
                # Check type
                expected_type = rules.get('type')
                if expected_type == 'string' and df[column].dtype != 'object':
                    errors.append(f"Column {column} should be string type")
                elif expected_type == 'integer' and not pd.api.types.is_integer_dtype(df[column]):
                    errors.append(f"Column {column} should be integer type")
                elif expected_type == 'float' and not pd.api.types.is_float_dtype(df[column]):
                    errors.append(f"Column {column} should be float type")
                
                # Check allowed values
                allowed_values = rules.get('allowed_values')
                if allowed_values:
                    invalid_values = set(df[column].unique()) - set(allowed_values)
                    if invalid_values:
                        errors.append(
                            f"Column {column} contains invalid values: {invalid_values}"
                        )
                
                # Check min/max values
                min_value = rules.get('min_value')
                if min_value is not None and df[column].min() < min_value:
                    errors.append(f"Column {column} has values below {min_value}")
                
                max_value = rules.get('max_value')
                if max_value is not None and df[column].max() > max_value:
                    errors.append(f"Column {column} has values above {max_value}")
            
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
            duplicate_pct = duplicates / len(df)
            
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
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
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