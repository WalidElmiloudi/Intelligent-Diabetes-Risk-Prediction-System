import pandas as pd
from pathlib import Path

path = "/opt/airflow"

def load_data():
    df = pd.read_csv(f"{path}/data/raw/patient_diagnosis.csv")
    df = df.drop(columns=["Unnamed: 0"])
    return df