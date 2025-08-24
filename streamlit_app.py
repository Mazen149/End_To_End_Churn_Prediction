import streamlit as st
import pandas as pd
import joblib
from utils.CustomerData import CustomerData
from utils.inference import predict_new
from utils.config import preprocessor, forest_model, xgboost_model

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("Customer Churn Prediction")
st.markdown("""
This application predicts whether a bank customer is likely to churn (leave the bank) based on various features.
""")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Input fields for customer data
    credit_score = st.slider("Credit Score")
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_credit_card = st.checkbox("Has Credit Card")
    is_active_member = st.checkbox("Is Active Member")
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

with col2:
    # Categorical inputs
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Create a button for prediction
if st.button("Predict Churn"):
    # Create CustomerData instance
    customer_data = CustomerData(
        CreditScore=credit_score,
        Geography=geography,
        Gender=gender,
        Age=age,
        Tenure=tenure,
        Balance=balance,
        NumOfProducts=num_products,
        HasCrCard=int(has_credit_card),
        IsActiveMember=int(is_active_member),
        EstimatedSalary=estimated_salary
    )
    
    # Make predictions using both models
    forest_result = predict_new(customer_data, preprocessor, forest_model)
    xgb_result = predict_new(customer_data, preprocessor, xgboost_model)
    
    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Random Forest Model:")
        st.write(f"Prediction: {'Churn' if forest_result['churn_prediction'] == 1 else 'No Churn'}")
        st.write(f"Probability: {forest_result['churn_probability']:.2%}")
    
    with col2:
        st.write("XGBoost Model:")
        st.write(f"Prediction: {'Churn' if xgb_result['churn_prediction'] == 1 else 'No Churn'}")
        st.write(f"Probability: {xgb_result['churn_probability']:.2%}")
