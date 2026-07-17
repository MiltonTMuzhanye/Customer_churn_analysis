# scripts/predict.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from src.customer_churn.pipelines.inference_pipeline import InferencePipeline
from src.customer_churn.pipelines.batch_prediction import BatchPredictor
from src.customer_churn.utils.logger import default_logger as logger
from src.customer_churn.utils.config import config_loader

def main():
    """Make predictions using the trained model."""
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file path or single customer data')
    parser.add_argument('--output', type=str, 
                       help='Output file path for predictions')
    parser.add_argument('--single', action='store_true',
                       help='Predict for a single customer')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting prediction")
        
        # Initialize pipeline
        inference_pipeline = InferencePipeline()
        batch_predictor = BatchPredictor(inference_pipeline)
        
        if args.single:
            # Single prediction
            df = pd.DataFrame([eval(args.input)]) if isinstance(args.input, str) else pd.DataFrame([args.input])
            results = inference_pipeline.predict(df)
            logger.info(f"Prediction results: {results['predictions'][0]}")
            print(results['predictions'][0])
        else:
            # Batch prediction
            results = batch_predictor.predict_file(args.input, args.output)
            logger.info(f"Batch prediction completed. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()