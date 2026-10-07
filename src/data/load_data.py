import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parents[2]

def load_data():
    df = pd.read_csv(f"{path}/data/raw/patient_diagnosis.csv")
    df = df.drop(columns=["Unnamed: 0"])
    return df