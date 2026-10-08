import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.customer_churn.pipelines.inference_pipeline import InferencePipeline
from src.customer_churn.pipelines.batch_prediction import BatchPredictor
from src.customer_churn.utils.logger import default_logger as logger


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer churn"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file path, or a JSON object when using --single",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path for batch predictions",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Predict one customer from a JSON object",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting prediction")
        pipeline = InferencePipeline()

        if args.single:
            try:
                customer = json.loads(args.input)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "--single requires valid JSON. "
                    "Pass a JSON object containing customer features."
                ) from exc

            if not isinstance(customer, dict):
                raise ValueError(
                    "Single-customer input must be a JSON object."
                )

            result = pipeline.predict(pd.DataFrame([customer]))
            print(json.dumps(result["predictions"][0], indent=2))

        else:
            input_path = Path(args.input)
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Input CSV not found: {input_path}"
                )

            output_path = args.output or (
                "data/processed/predictions/"
                f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )

            batch_predictor = BatchPredictor(pipeline)
            results = batch_predictor.predict_file(
                str(input_path),
                output_path,
            )

            print(f"Predictions generated: {len(results)}")
            print(f"Output saved to: {output_path}")

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
