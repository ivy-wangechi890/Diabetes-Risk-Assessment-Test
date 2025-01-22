import streamlit as st
import joblib 
import numpy as np
from sklearn.ensemble import  GradientBoostingClassifier


@st.cache_resource
def load_model():
    return joblib.load('gb.pkl')

st.cache_resource.clear()

st.title('Diabetes Risk Prediction Tool')
st.write(
    """
    This website is designed to help you understand your risk of developing diabetes.
    By analyzing key health and lifestyle factors, it provides valuable insights to support early detection 
    and preventive healthcare strategies.
    
    Take control of your health today!
    """
)

model=load_model()

if model:
    st.header("Please Enter The Following Details")

heart_disease = st.selectbox("Has heart disease?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
heart_disease_value = heart_disease[0]  # Extract the numeric value

hypertension = st.selectbox("Has hypertension?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
hypertension_value = hypertension[0]  # Extract the numeric value

age = st.number_input("Age", min_value=0, max_value=100, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Use only numeric values
input_data = np.array([age, hypertension_value, heart_disease_value, bmi, HbA1c_level, blood_glucose_level])

# Pass a unique key to the button
if st.button("Predict Diabetes Status", key="predict_button"):
    prediction = model.predict(input_data.reshape(1, -1))  # Ensure the input is in the correct shape
    diabetes_status = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"Predicted Diabetes Status: {diabetes_status}")

st.write('Actionable Recommendations: Use the results to make informed decisions about your health.')
    