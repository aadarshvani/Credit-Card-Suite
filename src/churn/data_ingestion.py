import os
import pandas as pd
import yaml
from src.utils.logger import get_logger

def load_params(params_path='params.yaml', section='churn_data_ingestion'):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params[section]['source_url']

def main():
    logger = get_logger('churn_data_ingestion')
    url = load_params()
    df = pd.read_csv(url)
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    os.makedirs('data/churn/raw', exist_ok=True)
    df.to_csv('data/churn/raw/cleaned.csv', index=False)
    logger.info(f"Saved cleaned data to data/churn/raw/cleaned.csv, shape: {df.shape}")

if __name__ == '__main__':
    main()
