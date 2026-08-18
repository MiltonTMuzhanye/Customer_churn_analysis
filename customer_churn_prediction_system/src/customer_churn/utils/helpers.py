import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import Any, Dict, List, Union
from datetime import datetime
import hashlib

def save_artifact(obj: Any, path: str):
    """Save an artifact to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_parquet(path)
    elif isinstance(obj, dict):
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        joblib.dump(obj, path)

def load_artifact(path: str) -> Any:
    """Load an artifact from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return joblib.load(path)

def create_features_hash(df: pd.DataFrame, columns: List[str] = None) -> str:
    """Create a hash of the feature set for versioning."""
    if columns is None:
        columns = df.columns
    data_string = ''.join(sorted(columns)) + str(df.shape)
    return hashlib.md5(data_string.encode()).hexdigest()

def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    class_counts = y.value_counts()
    total = len(y)
    weights = {
        cls: total / (len(class_counts) * count) 
        for cls, count in class_counts.items()
    }
    return weights

def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_bool_string(value: str) -> bool:
    """Parse boolean strings to boolean values."""
    if isinstance(value, bool):
        return value
    return value.lower() in ['true', '1', 't', 'yes', 'y']

def chunk_dataframe(df: pd.DataFrame, chunk_size: int):
    """Yield chunks of a dataframe."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]