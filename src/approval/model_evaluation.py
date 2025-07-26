import pickle
import pandas as pd
import json
import os
from sklearn.metrics import classification_report, recall_score
from src.utils.logger import get_logger

MODEL_PATH = 'models/approval.pkl'
X_TEST_PATH = 'data/approval/features/X_test.csv'
Y_TEST_PATH = 'data/approval/features/y_test.csv'
METRICS_PATH = 'metrics/approval_detection_metrics.json'

logger = get_logger('approval_evaluation')

def main():
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        # Load test data
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
        # Predict
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f'Recall: {recall:.4f}')
        logger.info('Classification Report:')
        logger.info(json.dumps(report, indent=2))
        # Write metrics for DVC
        with open(METRICS_PATH, 'w') as f:
            json.dump({'recall': recall, 'classification_report': report}, f, indent=4)
        print(f'\nRecall Score: {recall:.4f}\n')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
    except Exception as e:
        logger.error(f'Error in evaluation: {e}')
        raise

if __name__ == '__main__':
    main()