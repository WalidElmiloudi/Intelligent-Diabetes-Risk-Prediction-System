import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

path = Path(__file__).resolve().parents[2]

df = pd.read_csv(f"{path}/data/processed/cleaned_data.csv")

def train_kmeans():
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df_scaled)
    df["cluster"] = kmeans.predict(df_scaled)
    return df