from fastapi import FastAPI
import pandas as pd
from .schemas import Features
from .dependencies import get_model
from .services.prediction_service import predict

model = get_model()
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Diabetes Risk API"}

@app.get("/predict")
def predict(data : Features):
    df = pd.DataFrame([data.model_dump()])
    prediction = predict(model,df)
    prediction_value = float(prediction[0])
    return {
        "prediction": prediction_value
    }
