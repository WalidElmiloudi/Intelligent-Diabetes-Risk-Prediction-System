import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from src.classification.train import train
from pathlib import Path
from src.mlflow.tracking import start_experiment , log_parameters ,  log_model , log_metrics
import mlflow

path = Path(__file__).resolve().parents[2]

df = pd.read_csv(f"{path}/data/processed/interpreted_data.csv")

X = df.drop(columns="Diabete_risk")
y = df["Diabete_risk"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

trained_models = train()

def evaluate():
    accuracy = 0
    best_model = {}

    for name,model in trained_models.items():
        with mlflow.start_run(run_name=name):
            prediction = model.predict(X_test)

            log_parameters({
                "model":name,
            })

            report = classification_report(
                y_test,
                prediction,
                output_dict=True
            )
            metrics = {
                "accuracy": report["accuracy"],
                "high_precision": report["High"]["precision"],
                "high_recall": report["High"]["recall"],
                "high_f1": report["High"]["f1-score"],
                "low_precision": report["Low"]["precision"],
                "low_recall": report["Low"]["recall"],
                "low_f1": report["Low"]["f1-score"],
                "moderate_precision": report["Moderate"]["precision"],
                "moderate_recall": report["Moderate"]["recall"],
                "moderate_f1": report["Moderate"]["f1-score"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"]
            }
            log_metrics(metrics)

            log_model(name,model)

        print("===================================")
        print("===================================")
        print("Model:", name)
        print("Accuracy:", accuracy_score(y_test, prediction))

        print("\nClassification Report:")
        print(classification_report(y_test, prediction))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, prediction))
        print("\n")

        if(accuracy_score(y_test, prediction) > accuracy):
            accuracy = accuracy_score(y_test, prediction)
            best_model[name]=model

    return best_model

