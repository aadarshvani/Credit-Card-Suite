import mlflow
from contextlib import contextmanager

@contextmanager
def start_mlflow_run(run_name):
    mlflow.set_tracking_uri("mlruns")  # local directory
    mlflow.set_experiment("Churn_Prediction")
    with mlflow.start_run(run_name=run_name):
        yield
