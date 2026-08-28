import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.customer_churn.data.ingestion import DataIngestor
from src.customer_churn.data.validation import DataValidator
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

def main():
    """Ingest and validate data."""
    try:
        logger.info("Starting data ingestion")
        
        # Load config
        config = config_loader.get_config("config")
        data_config = config.get("data", {})
        
        # Initialize components
        ingestor = DataIngestor(config)
        validator = DataValidator()
        
        # Load data
        df = ingestor.load_data()
        
        # Validate data
        valid, summary = validator.validate_all(df)
        
        if not valid:
            logger.warning("Data validation failed. Proceeding with warnings.")
            logger.warning(f"Validation summary: {summary}")
        
        # Save processed data
        processed_path = data_config.get("processed_path", "data/processed/processed_data.parquet")
        ingestor.save_processed_data(df, processed_path)
        
        # Log information
        info = ingestor.get_data_info(df)
        logger.info(f"Data ingestion completed. Shape: {info['shape']}")
        logger.info(f"Memory usage: {info['memory_usage']:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()