# src/customer_churn/pipelines/batch_prediction.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from ..utils.logger import default_logger as logger
from ..utils.config import config_loader
from ..utils.helpers import chunk_dataframe

class BatchPredictor:
    """Handle batch prediction jobs."""
    
    def __init__(self, inference_pipeline):
        self.inference_pipeline = inference_pipeline
        self.config = config_loader.get_config("config")
        self.batch_size = self.config.get("inference", {}).get("batch_size", 1000)
        
    def predict_file(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """Make predictions on a file."""
        try:
            start_time = time.time()
            
            # Load data
            logger.info(f"Loading data from {input_path}")
            df = pd.read_csv(input_path)
            
            # Get predictions
            results = self.inference_pipeline.predict(df)
            
            # Format results
            predictions_df = pd.DataFrame(results['predictions'])
            
            # Add original data
            predictions_df = pd.concat([df, predictions_df], axis=1)
            
            # Save results
            output_path = output_path or f"data/processed/predictions/predictions_{int(time.time())}.csv"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(output_path, index=False)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Batch prediction completed in {elapsed_time:.2f} seconds")
            logger.info(f"Results saved to {output_path}")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise
    
    def predict_database(self, query: str, connection, output_path: str = None) -> pd.DataFrame:
        """Make predictions on database data."""
        try:
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, connection)
            
            results = self.predict_file(df, output_path)
            return results
            
        except Exception as e:
            logger.error(f"Database prediction error: {str(e)}")
            raise
    
    def stream_predict(self, df_generator, output_path: str = None):
        """Stream predictions from a generator."""
        try:
            results = []
            for chunk in df_generator:
                chunk_results = self.inference_pipeline.predict(chunk)
                results.extend(chunk_results['predictions'])
            
            predictions_df = pd.DataFrame(results)
            
            if output_path:
                predictions_df.to_csv(output_path, index=False)
                logger.info(f"Stream predictions saved to {output_path}")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Stream prediction error: {str(e)}")
            raise