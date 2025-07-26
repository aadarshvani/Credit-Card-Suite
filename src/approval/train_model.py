import os
import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from src.utils.logger import get_logger
import yaml

logger = get_logger("approval_train")
logger.info("Starting Approval Model Training Stage")

# ========== Load Data ==========
def load_data():
    X_train = pd.read_csv("data/approval/features/X_train.csv")
    y_train = pd.read_csv("data/approval/features/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/approval/features/X_test.csv")
    y_test = pd.read_csv("data/approval/features/y_test.csv").values.ravel()
    logger.info("Training and test data loaded.")
    return X_train, y_train, X_test, y_test

def load_params(path='params/approval.yaml'):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)['train_model']
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise

def main():
    logger = get_logger("approval_train")
    logger.info("Starting model training for approval prediction...")
    params = load_params()
    X_train = pd.read_csv("data/approval/features/X_train.csv")
    y_train = pd.read_csv("data/approval/features/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/approval/features/X_test.csv")
    y_test = pd.read_csv("data/approval/features/y_test.csv").values.ravel()
    logger.info("Loaded train and test features/labels.")
    # Logistic Regression
    log_reg_params = params['models']['logistic_regression']
    log_reg = LogisticRegression(**log_reg_params)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    recall_log = recall_score(y_test, y_pred_log)
    # Random Forest
    rf_params = params['models']['random_forest']
    rf_clf = RandomForestClassifier(**rf_params)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    recall_rf = recall_score(y_test, y_pred_rf)
    # Model selection
    if recall_log > recall_rf:
        chosen_model = log_reg
        y_pred_chosen = y_pred_log
        model_name = 'Logistic Regression'
        recall = recall_log
    else:
        chosen_model = rf_clf
        y_pred_chosen = y_pred_rf
        model_name = 'Random Forest Classifier'
        recall = recall_rf
    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/approval.pkl", "wb") as f:
        pickle.dump(chosen_model, f)
    logger.info(f"\nChosen Model Based on Recall: {model_name}")
    logger.info(f"Recall Score: {recall:.4f}")
    logger.info("Classification Report:")
    logger.info(json.dumps(classification_report(y_test, y_pred_chosen, output_dict=True), indent=2))
    print(f"\n📌 Chosen Model Based on Recall: {model_name}")
    print(f"🔁 Recall Score: {recall:.4f}\n")
    print("📈 Classification Report:")
    print(classification_report(y_test, y_pred_chosen))

if __name__ == "__main__":
    main()
