import streamlit as st
import joblib
import numpy as np

st.title("🎓 Student Performance Predictor")

@st.cache_resource
def load_model():
    model = joblib.load("student_score_model.pkl")
    return model

model = load_model()

st.write("Enter student study information")

study_hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)
previous_score = st.slider("Previous Score", 0, 100, 60)

input_data = np.array([[study_hours, attendance, sleep_hours, previous_score]])

if st.button("Predict Score"):
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Score: {prediction[0]:.2f}")
