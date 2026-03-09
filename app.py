import streamlit as st
import joblib
import pandas as pd

st.title("🎓 Student Performance Predictor")

@st.cache_resource
def load_artifacts():
    model = joblib.load("student_score_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, features = load_artifacts()

st.write("Enter student information")

study_hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)
previous_score = st.slider("Previous Score", 0, 100, 60)

# สร้าง dictionary ที่มี column ครบ
input_dict = {f: 0 for f in features}

# ใส่ค่าที่ผู้ใช้เลือก
if "StudyHours" in input_dict:
    input_dict["StudyHours"] = study_hours

if "Attendance" in input_dict:
    input_dict["Attendance"] = attendance

if "SleepHours" in input_dict:
    input_dict["SleepHours"] = sleep_hours

if "PreviousScore" in input_dict:
    input_dict["PreviousScore"] = previous_score

input_df = pd.DataFrame([input_dict])

if st.button("Predict Score"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Score: {prediction[0]:.2f}")
