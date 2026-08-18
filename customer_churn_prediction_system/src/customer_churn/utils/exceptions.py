class CustomerChurnError(Exception):
    """Base exception for customer churn prediction system."""
    pass

class DataValidationError(CustomerChurnError):
    """Exception raised for data validation errors."""
    pass

class DataIngestionError(CustomerChurnError):
    """Exception raised for data ingestion errors."""
    pass

class FeatureEngineeringError(CustomerChurnError):
    """Exception raised for feature engineering errors."""
    pass

class ModelTrainingError(CustomerChurnError):
    """Exception raised for model training errors."""
    pass

class ModelPredictionError(CustomerChurnError):
    """Exception raised for model prediction errors."""
    pass

class ConfigurationError(CustomerChurnError):
    """Exception raised for configuration errors."""
    pass