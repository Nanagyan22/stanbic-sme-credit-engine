import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---Page Config & State---
st.set_page_config(page_title="Stanbic SME Credit Engine", page_icon="🏦", layout="wide")
st.title("🏦 Stanbic Bank: Automated SME Credit Decisioning")
st.markdown("Prototype Pipeline: **XGBoost Classifier + Business Rules Engine**")

# ---Load the Serialized Pipeline & Model---
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('sme_preprocessor.pkl')
    model = joblib.load('sme_xgboost_model.pkl')
    return preprocessor, model

preprocessor, model = load_artifacts()

# ---Sidebar for Input---
st.sidebar.header("Applicant Details")
st.sidebar.markdown("Enter SME data to simulate inference:")

# Group 1: Business Details
st.sidebar.subheader("Business Profile")
sector = st.sidebar.selectbox("Sector", ['agriculture', 'manufacturing', 'retail/trading', 'hospitality', 'services', 'ict', 'construction', 'transport', 'healthcare', 'education', 'fishing', 'mining/quarrying'])
region = st.sidebar.selectbox("Region", ['greater accra', 'ashanti', 'western', 'eastern', 'central', 'northern', 'volta', 'brong ahafo', 'upper east', 'upper west'])
years_in_operation = st.sidebar.number_input("Years in Operation", min_value=0.0, max_value=50.0, value=3.0)
num_employees = st.sidebar.number_input("Number of Employees", min_value=1, value=5)

# Group 2: Financials
st.sidebar.subheader("Financials")
annual_revenue_ghs = st.sidebar.number_input("Annual Revenue (GHS)", min_value=0.0, value=120000.0)
avg_monthly_bank_balance_ghs = st.sidebar.number_input("Avg Monthly Balance (GHS)", min_value=0.0, value=15000.0)
has_momo_account = st.sidebar.selectbox("Has MoMo Account?", ['yes', 'no'])
monthly_momo_volume_ghs = st.sidebar.number_input("Monthly MoMo Volume (GHS)", min_value=0.0, value=5000.0) if has_momo_account == 'yes' else np.nan

# Group 3: Loan Request
st.sidebar.subheader("Loan Request")
loan_amount_requested_ghs = st.sidebar.number_input("Loan Amount Requested (GHS)", min_value=1000.0, value=60000.0)
loan_purpose = st.sidebar.selectbox("Loan Purpose", ['working capital', 'equipment', 'inventory purchase', 'business expansion', 'debt refinancing', 'vehicle purchase', 'asset purchase'])
collateral_type = st.sidebar.selectbox("Collateral Type", ['property', 'vehicle', 'equipment', 'cash deposit', 'guarantor only', 'none'])

# Background calculations - THE REAL-WORLD NEUTRAL BASELINE
input_data = pd.DataFrame({
    'sector': [sector], 'region': [region], 'years_in_operation': [years_in_operation], 
    'num_employees': [num_employees], 'annual_revenue_ghs': [annual_revenue_ghs],
    'monthly_momo_volume_ghs': [monthly_momo_volume_ghs], 'avg_monthly_bank_balance_ghs': [avg_monthly_bank_balance_ghs],
    'bank_account_tenure_months': [24], 
    'has_momo_account': [has_momo_account],
    'loan_amount_requested_ghs': [loan_amount_requested_ghs], 'loan_purpose': [loan_purpose],
    'collateral_type': [collateral_type], 
    'collateral_value_ghs': [loan_amount_requested_ghs * 0.8], # 80% collateral coverage
    'previous_loan_count': [1], 
    'previous_default': ['no'], 
    'credit_bureau_score': [None], 
    'rm_recommendation': ['refer'], # Neutral recommendation
    'internal_risk_grade': ['c'] # Average risk grade
})

# ---Prediction Engine---
if st.button("Run Credit Assessment Pipeline", type="primary"):
    with st.spinner("Processing application via ML pipeline..."):
        X_processed = preprocessor.transform(input_data)
        prob_default = model.predict_proba(X_processed)[0][1]
        
        # Apply Business Decision Engine
        if prob_default >= 0.40:
            decision = 'DECLINE'
            color = 'error'
        elif prob_default <= 0.15:
            decision = 'APPROVE'
            color = 'success'
        else:
            decision = 'REFER TO HUMAN'
            color = 'warning'
            
        # Hard Rule Override
        if decision == 'APPROVE' and years_in_operation < 1.0:
            decision = 'REFER TO HUMAN (Startup Override)'
            color = 'warning'

        # ---Display Results---
        st.divider()
        st.subheader("System Recommendation")
        
        if color == 'success':
            st.success(f"**{decision}**")
        elif color == 'error':
            st.error(f"**{decision}**")
        else:
            st.warning(f"**{decision}**")
            
        col1, col2 = st.columns(2)
        col1.metric("Predicted Probability of Default", f"{prob_default:.1%}")
        
        if decision == 'REFER TO HUMAN (Startup Override)':
            col2.info("📝 **Override Triggered:** Model approved, but business is < 1 year old. Routing to manual underwriter.")