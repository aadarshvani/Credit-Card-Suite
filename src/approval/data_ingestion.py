import os
import pandas as pd
import yaml
from src.utils.logger import get_logger

def load_params(params_path='params.yaml', section='approval_data_ingestion'):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params[section]['application_record_path'], params[section]['credit_record_path']

def main():
    logger = get_logger('approval_data_ingestion')
    app_url, credit_url = load_params()
    app_df = pd.read_csv(app_url)
    credit_df = pd.read_csv(credit_url)
    merged_df = pd.merge(app_df, credit_df, on='ID')
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.fillna(merged_df.median(numeric_only=True))
    os.makedirs('data/approval/raw', exist_ok=True)
    merged_df.to_csv('data/approval/raw/cleaned.csv', index=False)
    logger.info(f"Saved cleaned data to data/approval/raw/cleaned.csv, shape: {merged_df.shape}")

if __name__ == '__main__':
    main()