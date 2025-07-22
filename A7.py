import streamlit as st
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Glass Type Predictor", page_icon="🔮")

# Title
st.title("Glass Type Predictor")
st.markdown("Enter the glass chemical properties below to predict its type.")

# --- Load Model and Scaler ---
try:
    model = joblib.load("glass_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("Could not load model or scaler. Make sure 'glass_model.pkl' and 'scaler.pkl' are in the same folder as this script.")
    st.stop()

# --- Glass Type Mapping ---
glass_types = {
    1: "Building Windows Float Processed",
    2: "Building Windows Non-Float Processed",
    3: "Vehicle Windows Float Processed",
    4: "Vehicle Windows Non-Float Processed",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

# --- Input Form ---
with st.form("glass_form"):
    RI = st.number_input("Refractive Index (RI)", value=1.52)
    Na = st.number_input("Sodium (Na)", value=13.0)
    Mg = st.number_input("Magnesium (Mg)", value=2.0)
    Al = st.number_input("Aluminum (Al)", value=1.0)
    Si = st.number_input("Silicon (Si)", value=72.0)
    K  = st.number_input("Potassium (K)", value=0.5)
    Ca = st.number_input("Calcium (Ca)", value=8.0)
    Ba = st.number_input("Barium (Ba)", value=0.1)
    Fe = st.number_input("Iron (Fe)", value=0.1)

    submit = st.form_submit_button("Predict Glass Type")

# --- Prediction ---
if submit:
    try:
        features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
        scaled_input = scaler.transform(features)
        prediction = model.predict(scaled_input)[0]
        result = glass_types.get(prediction, "Unknown")
        st.success(f" Predicted Glass Type: **{result}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
