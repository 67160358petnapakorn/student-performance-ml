import streamlit as st
import pandas as pd
import joblib

st.title("Student Exam Score Prediction")

model = joblib.load("student_score_model.pkl")
features = joblib.load("features.pkl")

input_data = {}

for feature in features:
    value = st.number_input(feature, value=0.0)
    input_data[feature] = value

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("Predicted Exam Score:", prediction[0])
