import mlflow

def register_model(model_uri,name):
    mlflow.register_model(
        model_uri=model_uri,
        name=name
    )