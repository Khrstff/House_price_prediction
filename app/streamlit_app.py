import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.linear_model import LinearRegression

# Rutas absolutas
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "linear_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")

# Cargar modelo y scaler
model: LinearRegression = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Variables numéricas usadas en el entrenamiento
feature_names = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    # dummies (pueden variar si cambias el one-hot)
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]

# Título
st.title("🏡 Predicción de precio de vivienda")

# Inputs del usuario
st.subheader("📋 Características de la vivienda:")

input_data = {}

# Entradas numéricas
input_data["longitude"] = st.number_input("Longitud", value=-122.23)
input_data["latitude"] = st.number_input("Latitud", value=37.88)
input_data["housing_median_age"] = st.number_input("Antigüedad de la casa (años)", value=30)
input_data["total_rooms"] = st.number_input("Total de habitaciones", value=500)
input_data["total_bedrooms"] = st.number_input("Total de dormitorios", value=100)
input_data["population"] = st.number_input("Población en la zona", value=1000)
input_data["households"] = st.number_input("Número de hogares", value=300)
input_data["median_income"] = st.number_input("Ingreso medio", value=4.0)

# Dummy: ocean_proximity
ocean_choice = st.selectbox("Proximidad al océano", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])
for cat in ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]:
    input_data[f"ocean_proximity_{cat}"] = 1 if ocean_choice == cat else 0

# Convertir a DataFrame
input_df = pd.DataFrame([input_data])[feature_names]

# Escalar
scaled_input = scaler.transform(input_df)

# Predicción
if st.button("Predecir precio"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💲 Precio estimado: ${prediction:,.2f} USD")

