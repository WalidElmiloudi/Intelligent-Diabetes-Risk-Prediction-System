import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src.data.load_data import load_data
from pathlib import Path

path = Path(__file__).resolve().parents[2]

df = load_data()

df["Glucose"] = np.where(df["Glucose"] == 0,np.nan,df["Glucose"])
df["BloodPressure"] = np.where(df["BloodPressure"] == 0,np.nan,df["BloodPressure"])
df["SkinThickness"] = np.where(df["SkinThickness"] == 0,np.nan,df["SkinThickness"])
df["Insulin"] = np.where(df["Insulin"] == 0,np.nan,df["Insulin"])
df["BMI"] = np.where(df["BMI"] == 0,np.nan,df["BMI"])

selected_columns = ["SkinThickness","Insulin","BMI"]

for col in selected_columns :
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] > upper_bound,upper_bound,df[col])
    df[col] = np.where(df[col] < lower_bound,lower_bound,df[col])

imputer = KNNImputer(n_neighbors=5)

df[df.columns] = imputer.fit_transform(df)

df["Pregnancies"] = df["Pregnancies"].astype("Int64")
df["Age"] = df["Age"].astype("Int64")

df.to_csv(f"{path}/data/processed/cleaned_data.csv",index=False)