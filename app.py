# diabetes_gui.py

import streamlit as st
import numpy as np
import joblib

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Diabetes Prediction (Logistic Regression)",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Title & Intro ---
st.title("🩺 Diabetes Prediction App")
st.write("""
### Using Logistic Regression 
This app predicts whether a person is **Diabetic** or **Non-Diabetic**  
based on the **PIMA Diabetes Dataset**.
""")

# --- Load Your Logistic Regression Model ---
try:
    model = joblib.load("logistic_regression_model2.pkl")
except FileNotFoundError:
    st.error("❌ logistic_regression_model.pkl not found. Please place it in the same folder as this script.")
    st.stop()

# --- Model Info Sidebar ---
MODEL_NAME = "Logistic Regression"
MODEL_ACCURACY = 82.467532  # 🔸 replace this with your actual accuracy (float)

st.sidebar.header("📊 Model Information")
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write(f"**Accuracy:** {MODEL_ACCURACY:.4f}")
st.sidebar.markdown("---")
st.sidebar.info("Enter patient details on the right → then click **Predict Diabetes**")

# --- User Input Section ---
st.subheader("⚕️ Enter Patient Data")

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 120)
with col2:
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
with col3:
    insulin = st.number_input("Insulin Level", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0)

dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 33)

# --- Prepare Input ---
user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

# --- Predict Button ---
if st.button("🔍 Predict Diabetes"):
    prediction = model.predict(user_data)[0]

    if prediction == 1:
        st.error("🩸 The model predicts that this person is **Diabetic**.")
    else:
        st.success("💚 The model predicts that this person is **Non-Diabetic**.")

    st.markdown("---")
    st.write("✅ *Prediction completed using your trained Logistic Regression model.*")

# --- Footer ---
st.markdown("---")
st.caption("Developed by **Harshita Shakya** | Diabetes Prediction using Logistic Regression (PIMA Dataset)")

