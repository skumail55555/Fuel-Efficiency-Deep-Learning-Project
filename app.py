import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("ann_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# App title
st.title("Fuel Efficiency Prediction App 🚗")

st.write("Enter car details below:")

# User inputs
cylinders = st.slider("Cylinders", 2, 12, 4)

displacement = st.number_input(
    "Displacement",
    value=140.0
)

horsepower = st.number_input(
    "Horsepower",
    value=90.0
)

weight = st.number_input(
    "Weight",
    value=2500.0
)

# Prediction button
if st.button("Predict MPG"):

    input_data = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted MPG: {prediction[0][0]:.2f}"
    )