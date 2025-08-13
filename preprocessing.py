# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def preprocess(df):
    """Clean and encode dataset."""
    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')

    # Fill missing values with a placeholder
    df = df.fillna("Unknown")

    # Encode categorical (text/boolean) columns
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df
