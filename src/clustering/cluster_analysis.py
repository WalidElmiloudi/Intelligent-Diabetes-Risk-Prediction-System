import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.clustering.kmeans import train_kmeans
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[2]

df,kmeans,df_scaled = train_kmeans()

print(df["cluster"].value_counts())
feature_means = df.groupby("cluster").mean()
print("\n")
print(feature_means)

clusters = kmeans.fit_predict(df_scaled)

pca = PCA(n_components=2)

x_pca = pca.fit_transform(df_scaled)

plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=clusters
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Patient Clusters using PCA")

plt.savefig(f"{path}/src/clustering/Patient_clusters_using_pca.png")
plt.close()

df["Diabete_risk"] = np.where(df["cluster"] == 0 , "High" , np.where(df["cluster"] == 1 , "Moderate" , "Low"))
df = df.drop(columns=["cluster"])

df.to_csv(f"{path}/data/processed/interpreted_data.csv",index=False)