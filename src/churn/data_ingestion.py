import os
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ========== Logging Setup ==========
logger = logging.getLogger('data_pipeline')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ========== Load YAML Parameters ==========
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file).get('data_ingestion', {})
            if not params:
                raise KeyError("Missing 'data_ingestion' section in YAML file.")
            return params
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        exit(1)

# ========== Read Data ==========
def read_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("Dataset is empty.")
        logger.info("CSV data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        exit(1)

# ========== Data Processing ==========
def process(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in ['RowNumber', 'CustomerId']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
                logger.info(f"Dropped column: {col}")
            else:
                logger.warning(f"Column not found: {col}")
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        exit(1)


# ========== Save Data ==========
def save_data(output_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        os.makedirs(output_path, exist_ok=True)
        train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
        logger.info(f"Train/Test data saved in: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        exit(1)

# ========== Main ==========
def main():
    params = load_params('params.yaml')
    df = read_csv(params['source_url'])
    df = process(df)

    try:
        X = df.drop('Exited', axis=1)
        y = df['Exited']
    except KeyError:
        logger.error("'Exited' column not found.")
        exit(1)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=42, stratify=y)
        logger.info(f"Data split done: Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
    except Exception as e:
        logger.error(f"Train-test split failed: {e}")
        exit(1)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    save_data(os.path.join('data/churn', 'raw'), train_data, test_data)

if __name__ == "__main__":
    main()
