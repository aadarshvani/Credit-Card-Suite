# 🚂 train_model.py (Formatted with logger, exception handling, CV, and DVC metrics)
import os
import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from src.utils.logger import get_logger

# --- Configurable file paths ---
PARAMS_PATH = 'params/churn.yaml'
X_TRAIN_PATH = 'data/churn/features/X_train.csv'
Y_TRAIN_PATH = 'data/churn/features/y_train.csv'
X_TEST_PATH = 'data/churn/features/X_test.csv'
Y_TEST_PATH = 'data/churn/features/y_test.csv'
MODEL_DIR = 'models'
METRICS_DIR = 'metrics'

logger = get_logger("churn_train")

# --- Load Parameters ---
def load_params(path=PARAMS_PATH):
    """Load model training parameters from YAML."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)['train_model']
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise

# --- Load Data ---
def load_data():
    """Load train and test features and labels."""
    try:
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
        logger.info("Loaded train and test features/labels.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load training/testing data: {e}")
        raise

# --- Evaluate Model ---
def evaluate(model, X_test, y_test):
    """Compute evaluation metrics for a model."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

# --- Main Training Logic ---
def main():
    logger = get_logger("churn_train")
    logger.info("Starting model training for churn prediction...")
    # Load data
    X_train = pd.read_csv("data/churn/features/X_train.csv")
    y_train = pd.read_csv("data/churn/features/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/churn/features/X_test.csv")
    y_test = pd.read_csv("data/churn/features/y_test.csv").values.ravel()
    logger.info("Loaded train and test features/labels.")
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    recall_log = recall_score(y_test, y_pred_log)
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
    with open("models/churn.pkl", "wb") as f:
        pickle.dump(chosen_model, f)
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "model": model_name,
        "recall": recall,
        "classification_report": classification_report(y_test, y_pred_chosen, output_dict=True)
    }
    with open("metrics/churn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"\nChosen Model Based on Recall: {model_name}")
    logger.info(f"Recall Score: {recall:.4f}")
    logger.info("Classification Report:")
    logger.info(json.dumps(metrics["classification_report"], indent=2))
    print(f"\n📌 Chosen Model Based on Recall: {model_name}")
    print(f"🔁 Recall Score: {recall:.4f}\n")
    print("📈 Classification Report:")
    print(classification_report(y_test, y_pred_chosen))

if __name__ == "__main__":
    main()
