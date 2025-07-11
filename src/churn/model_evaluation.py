import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.utils.logger import get_logger
from src.utils.mlflow_tracker import start_mlflow_run

import mlflow

logger = get_logger("churn_evaluation")

def load_data(x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel() 
    return X, y


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    logger.info("Classification Report:\n" + classification_report(y_test, preds))
    return metrics, preds, probs

def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

def save_roc_curve(y_true, y_prob, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_true, y_prob))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "roc_curve.png"))
    plt.close()


def log_metrics_mlflow(metrics: dict):
    with start_mlflow_run("Churn Evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("evaluation/churn/roc_curve.png")
        mlflow.log_artifact("evaluation/churn/confusion_matrix.png")


def main():
    logger.info("Starting Churn Model Evaluation...")

    # Load model
    model_path = "models/churn_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    X_test, y_test = load_data('data/churn/features/X_test.csv',
                                'data/churn/features/y_test.csv')

    # Evaluate
    metrics, preds, probs = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # Save plots
    output_path = "evaluation/churn"
    save_confusion_matrix(y_test, preds, output_path)
    save_roc_curve(y_test, probs, output_path)

    # Log to MLflow
    log_metrics_mlflow(metrics)

    # Save to DVC-readable metrics.json
    import json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logger.info("Churn Model Evaluation Completed.")

if __name__ == "__main__":
    main()
