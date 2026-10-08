import mlflow.sklearn

def get_model():
    model_uri = "models:/DiabetesRiskClassifier@champion"

    return mlflow.sklearn.load_model(model_uri)