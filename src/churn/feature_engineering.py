import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

def main():
    logger = get_logger('churn_feature_engineering')
    df = pd.read_csv('data/churn/raw/cleaned.csv')
    # Encode target
    if 'Attrition_Flag' in df.columns:
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
    # Simple one-hot encoding for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Fillna (should be none, but for safety)
    df = df.fillna(df.median(numeric_only=True))
    # Split features and target
    X = df.drop(columns=['Attrition_Flag'])
    y = df['Attrition_Flag']
    
    # Save feature names before scaling
    feature_names = X.columns.tolist()
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    # Scale features
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    # Save outputs
    os.makedirs('data/churn/features', exist_ok=True)
    pd.DataFrame(X_train).to_csv('data/churn/features/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/churn/features/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['Attrition_Flag']).to_csv('data/churn/features/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['Attrition_Flag']).to_csv('data/churn/features/y_test.csv', index=False)
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/churn_transformer.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # Save feature names for backend use
    with open('artifacts/churn_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info('Feature engineering complete and files saved.')

if __name__ == '__main__':
    main()
