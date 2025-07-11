import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger

logger = get_logger('feature_engineering')
logger.info("Starting Feature Engineering Stage!")

# ========== Load Data ==========
def load_data(train_path: str, test_path: str):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if train_df.empty or test_df.empty:
            raise ValueError("Train or Test file is empty.")

        logger.info("Raw train/test data loaded successfully.")
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        exit(1)

# ========== Build Transformer ==========
def build_transformer(X: pd.DataFrame) -> ColumnTransformer:
    try:
        cat_cols = ['Geography', 'Gender']
        num_cols = X.select_dtypes(include=['int64', 'float64']).drop(columns=cat_cols, errors='ignore').columns.tolist()

        transformer = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ])
        logger.info("ColumnTransformer defined.")
        return transformer

    except Exception as e:
        logger.error(f"Error building transformer: {e}")
        exit(1)

# ========== Apply Transformer ==========
def transform_features(transformer, X_train, X_test):
    try:
        X_train_encoded = transformer.fit_transform(X_train)
        X_test_encoded = transformer.transform(X_test)
        logger.info("Feature transformation completed.")
        return X_train_encoded, X_test_encoded
    
    except Exception as e:
        logger.error(f"Error transforming features: {e}")
        exit(1)

# ========== Apply SMOTE ==========
def apply_smote(X, y):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info("SMOTE applied to balance classes.")
        return X_resampled, y_resampled

    except Exception as e:
        logger.error(f"Error applying SMOTE: {e}")
        exit(1)

# ========== Save Outputs ==========
def save_outputs(X_train, y_train, X_test, y_test, transformer):
    try:
        os.makedirs('data/churn/features', exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)

        pd.DataFrame(X_train).to_csv('data/churn/features/X_train_resampled.csv', index=False)
        pd.DataFrame(y_train, columns=['Exited']).to_csv('data/churn/features/y_train_resampled.csv', index=False)
        pd.DataFrame(X_test).to_csv('data/churn/features/X_test.csv', index=False)
        pd.DataFrame(y_test, columns=['Exited']).to_csv('data/churn/features/y_test.csv', index=False)

        with open('artifacts/column_transformer.pkl', 'wb') as f:
            pickle.dump(transformer, f)

        logger.info("Feature engineered datasets and transformer saved.")
    
    except Exception as e:
        logger.error(f"Error saving outputs: {e}")
        exit(1)


# ========== Main ==========
def main():
    train_df, test_df = load_data('data//churn/raw/train.csv', 'data/churn/raw/test.csv')

    try:
        X_train = train_df.drop('Exited', axis=1)
        y_train = train_df['Exited']
        X_test = test_df.drop('Exited', axis=1)
        y_test = test_df['Exited']
    except KeyError:
        logger.error("'Exited' column not found in data.")
        exit(1)

    transformer = build_transformer(X_train)
    X_train_encoded, X_test_encoded = transform_features(transformer, X_train, X_test)
    X_train_resampled, y_train_resampled = apply_smote(X_train_encoded, y_train)
    
    save_outputs(X_train_resampled, y_train_resampled, X_test_encoded, y_test, transformer)


if __name__ == '__main__':
    main()
