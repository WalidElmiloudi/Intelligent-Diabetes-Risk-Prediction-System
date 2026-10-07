import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
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

pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num",pipeline,X.columns)
])

logistic_regression_model = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(
        solver='saga',
        max_iter=10000
    ))
])

knn_classifier_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

decision_tree_model = DecisionTreeClassifier()

random_forest_model = RandomForestClassifier()

svc = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC())
])

naive_bayes_model = GaussianNB()

gradient_boosting_model = GradientBoostingClassifier()

models = {
    "Logistic Regression": logistic_regression_model,
    "KNeighbors Classifier": knn_classifier_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "SVC": svc,
    "Naive Bayes": naive_bayes_model,
    "Gradient Boosting": gradient_boosting_model
}

def train():
    trained_models = {}
    for name,model in models.items():
        model.fit(X_train,y_train)
        trained_models[name] = model

    return trained_models
