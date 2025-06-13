import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model.predict import predict_emergency
from utils.preprocessing import preprocess_ecg

st.title("🫀 CardioGuard AI")
st.subheader("Early Detection of Cardiac Emergencies from ECG")

uploaded_file = st.file_uploader("Upload ECG file (.csv)", type=["csv"])
if uploaded_file:
    st.success("ECG data uploaded!")
    ecg_data = np.loadtxt(uploaded_file, delimiter=",")
    
    st.line_chart(ecg_data[:1000])  # Show first few points
    
    with st.spinner("Analyzing ECG..."):
        processed = preprocess_ecg(ecg_data)
        result = predict_emergency(processed)
    
    st.subheader("🧠 Prediction:")
    st.write(f"🔔 **{result['prediction']}**")
    st.write("Confidence:", result["confidence"])
