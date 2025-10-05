import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)
