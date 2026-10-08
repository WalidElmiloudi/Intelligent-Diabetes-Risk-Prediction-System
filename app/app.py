import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)

st.title("Diabetes Risk Prediction")

st.write(
    "Enter the patient information below to predict "
    "the estimated diabete risk."
)

st.subheader("Patient Information")

patient_pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=0
)

patient_glucose = st.number_input(
    "Glucose",
    min_value=40,
    max_value=200,
    value=44
)

patient_blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=22,
    max_value=122,
    value=24
)

patient_skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=5,
    max_value=57,
    value=7
)

patient_insulin = st.number_input(
    "Insulin",
    min_value=12.0,
    max_value=360.0,
    value=15.0,
    step=0.1
)

patient_bmi = st.number_input(
    "Body Mass Index (BMI)",
    min_value=18.0,
    max_value=50.25,
    value=20.0,
    step=0.1
)

patient_diabetes_pedigree_function = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.05,
    max_value=2.5,
    value=0.6
)

patient_age = st.number_input(
    "Age",
    min_value=18,
    max_value=120,
    value=21
)

if st.button("Predict Diabetes Risk"):

    data = {
        "Pregnancies": patient_pregnancies,
        "Glucose": patient_glucose,
        "BloodPressure": patient_blood_pressure,
        "SkinThickness": patient_skin_thickness,
        "Insulin": patient_insulin,
        "BMI": patient_bmi,
        "DiabetesPedigreeFunction": patient_diabetes_pedigree_function,
        "Age": patient_age
    }

    URL = "http://127.0.0.1:8000"

    try:
        response = requests.post(f"{URL}/predict",json=data)
        result = response.json()
    except requests.exceptions.ConnectionError:
        print("Could not connect to the server.")

    predicted_risk = result["prediction"]

    st.success(
        f"Diabetes Risk : **{predicted_risk}**"
    )
