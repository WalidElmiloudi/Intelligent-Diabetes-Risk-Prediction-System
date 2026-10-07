import mlflow
import mlflow.sklearn

MLFLOW_URI = "http://127.0.0.1:5000"


def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)

def start_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def log_parameters(params):
    for name,value in params.items():
        mlflow.log_param(name,value)

def log_metrics(metrics):
    for name,value in metrics.items():
        mlflow.log_metric(name,value)

def log_model(name,model):
    mlflow.sklearn.log_model(
        model,
        name=name,
        serialization_format="cloudpickle"
    )