import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main-title{
font-size:45px;
font-weight:700;
text-align:center;
}

.sub-title{
text-align:center;
color:gray;
margin-bottom:40px;
}

.card{
background:#111;
padding:25px;
border-radius:12px;
box-shadow:0 0 10px rgba(0,0,0,0.3);
}

.result{
padding:25px;
border-radius:12px;
background:linear-gradient(90deg,#2a9d8f,#1f7a8c);
text-align:center;
font-size:30px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Model for Predicting Student Exam Score</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("student_score_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, features = load_artifacts()

# ---------------- INPUT AREA ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📚 Study Information")

    study_hours = st.slider("Study Hours",0,10,5)
    attendance = st.slider("Attendance (%)",0,100,80)

with col2:
    st.markdown("### 🧠 Lifestyle")

    sleep_hours = st.slider("Sleep Hours",0,10,7)
    previous_score = st.slider("Previous Score",0,100,60)

# ---------------- PREPARE INPUT ----------------
input_dict = {f:0 for f in features}

if "StudyHours" in input_dict:
    input_dict["StudyHours"] = study_hours

if "Attendance" in input_dict:
    input_dict["Attendance"] = attendance

if "SleepHours" in input_dict:
    input_dict["SleepHours"] = sleep_hours

if "PreviousScore" in input_dict:
    input_dict["PreviousScore"] = previous_score

input_df = pd.DataFrame([input_dict])

st.markdown("")

# ---------------- PREDICT BUTTON ----------------
if st.button("🚀 Predict Student Score", use_container_width=True):

    prediction = model.predict(input_df)[0]

    st.markdown(
        f'<div class="result">📊 Predicted Score : {prediction:.2f}</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📈 Performance Meter")

    st.progress(int(prediction)/100)

    # simple chart
    chart_data = pd.DataFrame({
        "Metric":["Study","Attendance","Sleep","Previous"],
        "Value":[study_hours,attendance,sleep_hours,previous_score]
    })

    st.bar_chart(chart_data.set_index("Metric"))

st.markdown("---")
st.caption("Student ML Prediction App • Built with Streamlit")
