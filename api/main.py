from fastapi import FastAPI
import pandas as pd
from .schemas import Features
from .dependencies import get_model
from .services.prediction_service import predict_risk

model = get_model()
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Diabetes Risk API"}

@app.post("/predict")
def predict(data : Features):
    df = pd.DataFrame([data.model_dump()])
    prediction = predict_risk(model,df)
    prediction_value = str(prediction[0])
    return {
        "prediction": prediction_value
    }
