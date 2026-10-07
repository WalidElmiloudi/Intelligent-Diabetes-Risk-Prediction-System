import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from src.classification.train import train
from pathlib import Path

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
        prediction = model.predict(X_test)
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

