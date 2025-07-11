import os
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger
from src.utils.mlflow_tracker import start_mlflow_run

logger = get_logger("churn_train")

# ========== Load parameters ==========
def load_params(path='params/churn.yaml'):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)['train_model']
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        exit(1)

# ========== Load data ==========
def load_data(X_path: str, y_path: str):
    try:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        return X, y.values.ravel()  # flatten y if needed
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        exit(1)


# ========== Get Model Object ==========
def get_model_obj(name):
    if name == "logistic_regression":
        return LogisticRegression()
    elif name == "knn":
        return KNeighborsClassifier()
    elif name == "decision_tree":
        return DecisionTreeClassifier()
    elif name == "random_forest":
        return RandomForestClassifier()
    elif name == "gradient_boosting":
        return GradientBoostingClassifier()
    elif name == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model: {name}")

# ========== Evaluate ==========
def evaluate(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        'roc_auc': roc_auc_score(y, proba)
    }

# ========== Main Training ==========
def main():
    logger.info("Starting training...")

    # Load config and data
    params = load_params()
    X_train, y_train = load_data('data/churn/features/X_train_resampled.csv',
                                 'data/churn/features/y_train_resampled.csv')
    X_test, y_test = load_data('data/churn/features/X_test.csv',
                                'data/churn/features/y_test.csv')
    scoring = params['scoring']
    cv = params['cv']

    best_model = None
    best_score = -np.inf
    best_model_name = ''
    best_metrics = {}

    for name, hyperparams in params['models'].items():
        logger.info(f"Training: {name}")
        model = get_model_obj(name)
        clf = GridSearchCV(model, hyperparams, cv=cv, scoring=scoring, n_jobs=-1)
        clf.fit(X_train, y_train)

        train_metrics = evaluate(clf.best_estimator_, X_train, y_train)
        test_metrics = evaluate(clf.best_estimator_, X_test, y_test)

        logger.info(f"Best Params for {name}: {clf.best_params_}")
        logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")

        # Log to MLflow
        with start_mlflow_run(name):
            import mlflow
            mlflow.log_params(clf.best_params_)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.sklearn.log_model(clf.best_estimator_, "model")

        if test_metrics['roc_auc'] > best_score:
            best_score = test_metrics['roc_auc']
            best_model = clf.best_estimator_
            best_model_name = name
            best_metrics = test_metrics

    # Save best model using pickle
    os.makedirs("models", exist_ok=True)
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model: {best_model_name} with ROC AUC: {best_score:.4f}")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
