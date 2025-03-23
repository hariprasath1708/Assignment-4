import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and expected feature names
model = joblib.load("env/Scripts/employee_attrition_model (4).pkl")
training_columns = joblib.load("env/Scripts/training_columns.pkl")  # Load feature names

# Streamlit App Title
st.title("🔍 Employee Attrition Prediction")

# User Input Section
st.header("📋 Enter Employee Details")

# User Input Fields (Main Page)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
overtime = st.selectbox("Overtime", ["No", "Yes"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Manager", "Laboratory Technician", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

# Create an input DataFrame with the correct feature names
input_data = pd.DataFrame(columns=training_columns)  # Ensure all columns exist

# Fill with default values (0)
input_data.loc[0] = 0  

# Assign user inputs to the correct feature columns
input_data["Age"] = age
input_data["MonthlyIncome"] = monthly_income
input_data["JobSatisfaction"] = job_satisfaction
input_data["OverTime_Yes"] = 1 if overtime == "Yes" else 0

# One-hot encoding (Set the correct category to 1, others remain 0)
if f"Department_{department}" in training_columns:
    input_data[f"Department_{department}"] = 1
if f"EducationField_{education_field}" in training_columns:
    input_data[f"EducationField_{education_field}"] = 1
if f"MaritalStatus_{marital_status}" in training_columns:
    input_data[f"MaritalStatus_{marital_status}"] = 1
if f"JobRole_{job_role}" in training_columns:
    input_data[f"JobRole_{job_role}"] = 1
if f"Gender_{gender}" in training_columns:
    input_data[f"Gender_{gender}"] = 1
if f"BusinessTravel_{business_travel}" in training_columns:
    input_data[f"BusinessTravel_{business_travel}"] = 1

# Ensure input matches model's expected features
input_data = input_data[training_columns]  # Reorder columns

# Predict when button is clicked
if st.button("🔍 Predict Attrition"):
    prediction = model.predict(input_data)
    result = "🚀 Likely to Leave" if prediction[0] == 1 else "✅ Likely to Stay"
    
    # Show result on the main page
    st.header("🎯 Prediction Result")
    st.subheader(result)
