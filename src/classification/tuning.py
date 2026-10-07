import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from src.classification.evaluate import evaluate
import joblib
from pathlib import Path
from src.mlflow.tracking import start_experiment , log_parameters ,  log_model , log_metrics
import mlflow
import mlflow.sklearn

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

start_experiment("Diabetes Risk Classification - Tuning")

best_model = evaluate()

param_grid = [
    {
        'classifier__l1_ratio': [0],
        'classifier__C': np.logspace(-3, 3, 20),
    },
    {
        'classifier__l1_ratio': [1],
        'classifier__C': np.logspace(-3, 3, 20),
    },
    {
        'classifier__l1_ratio': [0.25, 0.5, 0.75],
        'classifier__C': np.logspace(-3, 3, 20),
    }
]

model = ""
for _,value in best_model.items():
    model = value

search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=5,
        n_iter=30,
        scoring="f1_macro",
        random_state=42,
        n_jobs=1
    )

search.fit(X_train,y_train)
print("Best parameters:", search.best_params_)
print("Best CV score:", search.best_score_ * 100)

best_model = search.best_estimator_

best_model.fit(X_train,y_train)

with mlflow.start_run(run_name="Best Model"):

    log_parameters(search.best_params_)
    log_metrics({
        "accuracy": search.best_score_
    })

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="diabetes_risk_model",
        registered_model_name="DiabetesRiskClassifier",
        serialization_format="cloudpickle"
    )

joblib.dump(best_model,f"{path}/models/classification/final_model.joblib")