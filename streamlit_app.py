import streamlit as st
import pandas as pd
import joblib
from utils.CustomerData import CustomerData
from utils.inference import predict_new
from utils.config import preprocessor, forest_model, xgboost_model

# Set page config and styling
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS to improve UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #1E88E5;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    div[data-testid="stHeader"] {
        background-color: #E3F2FD;
        padding: 1rem;
    }
    .recommendation-item {
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
        background-color: #ffffff;
        border-radius: 0 5px 5px 0;
    }
    .recommendation-header {
        color: #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    /* Style for progress bars */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    /* Style for slider progress */
    .stSlider div[data-baseweb="slider"] div[role="progressbar"] {
        background-color: #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with improved styling
st.title("🏦 Customer Churn Prediction")
st.markdown("""
<div style='background-color: #E3F2FD; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border: 2px solid #1E88E5;'>
    <h3 style='color: #1E88E5; text-align: center;'>Welcome to the Bank Customer Churn Predictor!</h3>
    <p style='text-align: center; color: #424242;'>This intelligent system analyzes customer data to predict the likelihood of a customer leaving the bank. 
    Simply fill in the customer information below, and our advanced machine learning models will provide insights about potential churn risk.</p>
</div>
""", unsafe_allow_html=True)

# Create a container for customer information
with st.container():
    st.markdown("### 📊 Enter Customer Details")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 📝 Basic Information")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                               help="Customer's country")
        gender = st.selectbox("Gender", ["Male", "Female"],
                            help="Customer's gender")
        age = st.slider("Age", 18, 100, 30, 
                       help="Customer's age in years (18-100)")
        
    with col2:
        st.markdown("#### 💳 Account Details")
        credit_score = st.slider("Credit Score", 0, 1000, 500,
                               help="Credit score of the customer (0-1000)")
        balance = st.number_input("Account Balance", 0.0, step=1000.0, value=50000.0,
                                help="Account balance (minimum: 0)")
        tenure = st.slider("Tenure (years with Bank)", 0, 10, 5,
                         help="Years as a customer (0-10)")

    with col3:
        st.markdown("#### 💼 Customer Profile")
        estimated_salary = st.number_input("Estimated Salary", 0.0, step=1000.0, value=50000.0,
                                         help="Estimated annual salary (minimum: 0)")
        is_active_member = st.checkbox("Active Member",
                                     help="Is the customer regularly using their account?")
        has_credit_card = st.checkbox("Has Credit Card", 
                                    help="Does the customer have a credit card?")
        num_products = st.slider("Number of Bank Products", 1, 4, 1,
                               help="Number of banking products used (1-4)")

# Add a separator
st.markdown("---")

# Create a button for prediction with custom styling
predict_button = st.button("Predict Customer Churn", use_container_width=True)

# Container for prediction results
results_container = st.container()

if predict_button:
    with st.spinner('Analyzing customer data...'):
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

    with results_container:
        st.markdown("### 🎯 Prediction Results")
        
        # Calculate average probability
        avg_prob = (forest_result['churn_probability'] + xgb_result['churn_probability']) / 2
        
        # Display overall risk level
        risk_color = "#ff4b4b" if avg_prob > 0.5 else "#00cc96"
        st.markdown(f"""
            <div style='background-color: {risk_color}; padding: 1rem; border-radius: 5px; color: white; text-align: center; margin-bottom: 2rem;'>
                <h2>Overall Churn Risk: {avg_prob:.1%}</h2>
                <p>{'High Risk ⚠️' if avg_prob > 0.5 else 'Low Risk ✅'}</p>
            </div>
        """, unsafe_allow_html=True)

        # Create two columns for model results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px;'>
                    <h4>Random Forest Model</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2 style='color: {"#ff4b4b" if forest_result["churn_prediction"] == 1 else "#00cc96"}'>
                        {'Will Churn' if forest_result["churn_prediction"] == 1 else 'Will Stay'}
                    </h2>
                    <p>Confidence: {forest_result["churn_probability"]:.1%}</p>
                </div>
                </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown("""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px;'>
                    <h4>XGBoost Model</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2 style='color: {"#ff4b4b" if xgb_result["churn_prediction"] == 1 else "#00cc96"}'>
                        {'Will Churn' if xgb_result["churn_prediction"] == 1 else 'Will Stay'}
                    </h2>
                    <p>Confidence: {xgb_result["churn_probability"]:.1%}</p>
                </div>
                </div>
            """, unsafe_allow_html=True)

        # Add a separator before recommendations
        st.markdown("---")
            
        # Add recommendations section with enhanced styling
        st.markdown("""
            <div class='recommendation-header'>
                📋 Action Plan & Recommendations
            </div>
        """, unsafe_allow_html=True)
        
        recommendations = []
        if credit_score < 20:
            recommendations.append({
                "icon": "💳",
                "text": "Help customer improve their credit score with basic financial tips.",
                "priority": "Medium"
            })
        if not is_active_member:
            recommendations.append({
                "icon": "🎯",
                "text": "Send special offers to bring customer back to active status.",
                "priority": "High"
            })
        if num_products == 1:
            recommendations.append({
                "icon": "📈",
                "text": "Suggest other helpful banking products to the customer.",
                "priority": "High"
            })
        if not has_credit_card:
            recommendations.append({
                "icon": "💰",
                "text": "Recommend our credit card with its best features.",
                "priority": "Low"
            })
        
        if recommendations:
            st.markdown("""
                <div style='background-color: #E3F2FD; padding: 1.5rem; border-radius: 10px; border: 2px solid #1E88E5;'>
                    <h4 style='color: #1E88E5; margin-bottom: 1rem; text-align: center;'>
                        Suggested Actions to Reduce Churn Risk
                    </h4>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                    <div class='recommendation-item'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 1.2rem; margin-right: 10px;'>{rec['icon']}</span>
                                <span>{rec['text']}</span>
                            </div>
                            <span style='color: {"#1E88E5" if rec["priority"] == "High" else "#42A5F5"}; 
                                       font-weight: bold;'>
                                {rec['priority']} Priority
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
