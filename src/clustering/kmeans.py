import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
from src.mlflow.tracking import start_experiment , log_parameters , log_model

path = Path(__file__).resolve().parents[2]

df = pd.read_csv(f"{path}/data/processed/cleaned_data.csv")

def train_kmeans():
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    start_experiment("Diabetes Risk Clustering")

    with mlflow.start_run(run_name="KMeans"):
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(df_scaled)

        log_parameters({
            "n_clusters":3,
            "random_state":42
        })

        log_model("KMeans",kmeans)


    df["cluster"] = kmeans.predict(df_scaled)
    return [df,kmeans,df_scaled]