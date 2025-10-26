import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Maternal Health Risk Prediction App")

st.markdown("""
### Predict the risk level of maternal health complications  
Provide patient details below:
""")

# Input fields
BS = st.number_input('Blood Sugar Level (mmol/L)', min_value=0.0)
SystolicBP = st.number_input('Systolic Blood Pressure (mm Hg)', min_value=0.0)
DiastolicBP = st.number_input('Diastolic Blood Pressure (mm Hg)', min_value=0.0)
Age = st.number_input('Age (years)', min_value=0.0)
MAP = st.number_input('Mean Arterial Pressure (mm Hg)', min_value=0.0)
HeartRate = st.number_input('Heart Rate (bpm)', min_value=0.0)
BodyTemp = st.number_input('Body Temperature (°F)', min_value=0.0)

# Predict button
if st.button("Predict Risk Level"):
    # Arrange data in same order as training
    new_data = np.array([[BS, SystolicBP, Age, MAP, HeartRate, DiastolicBP, BodyTemp]])
    
    # Scale input
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    prediction = xgb_model.predict(new_data_scaled)[0]
    probabilities = xgb_model.predict_proba(new_data_scaled)[0]
    
    # Map results
    risk_labels = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}
    predicted_label = risk_labels.get(prediction, "Unknown")
    predicted_prob = np.max(probabilities)

    # Display result
    st.subheader("Prediction Result")
    st.success(f"**Risk Level:** {predicted_label}")
    st.info(f"**Model Confidence:** {predicted_prob:.2f}")
