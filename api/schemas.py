from pydantic import BaseModel

class Features(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : float
    SkinThickness : float
    Insulin : float
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int