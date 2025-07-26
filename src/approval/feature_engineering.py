import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger
import yaml

def load_params(path='params/approval.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)['preprocessing']

def main():
    logger = get_logger('approval_feature_engineering')
    params = load_params()
    df = pd.read_csv('data/approval/raw/cleaned.csv')
    # Label transformation
    if 'STATUS' in df.columns:
        df['client_status'] = df['STATUS'].apply(lambda x: 1 if str(x) in ['2', '3', '4', '5'] else 0)
    else:
        logger.error('STATUS column not found.')
        return
    # Drop columns if present
    for col in params.get('drop_columns', []):
        if col in df.columns:
            df = df.drop(columns=col)
    # One-hot encode categorical variables
    if params.get('one_hot_encode', True):
        cat_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Fillna
    if params.get('fillna', 'median') == 'median':
        df = df.fillna(df.median(numeric_only=True))
    # Split features and target
    X = df.drop(columns=['client_status'])
    y = df['client_status']
    
    # Save feature names before scaling
    feature_names = X.columns.tolist()
    
    # Handle class imbalance
    smote_params = params.get('smote', {'random_state': 42})
    smote = SMOTE(**smote_params)
    X_res, y_res = smote.fit_resample(X, y)
    # Scale features
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    # Save outputs
    os.makedirs('data/approval/features', exist_ok=True)
    pd.DataFrame(X_train).to_csv('data/approval/features/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/approval/features/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['client_status']).to_csv('data/approval/features/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['client_status']).to_csv('data/approval/features/y_test.csv', index=False)
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/approval_transformer.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # Save feature names for backend use
    with open('artifacts/approval_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info('Feature engineering complete and files saved.')

if __name__ == '__main__':
    main()