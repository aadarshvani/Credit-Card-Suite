import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger

MODEL_PATH = 'models/churn.pkl'
X_TEST_PATH = 'data/churn/features/X_test.csv'
Y_TEST_PATH = 'data/churn/features/y_test.csv'

logger = get_logger('churn_evaluation')

def main():
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs)
    }
    for k, v in metrics.items():
        logger.info(f'{k}: {v:.4f}')
    print('Evaluation metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

if __name__ == '__main__':
    main()
