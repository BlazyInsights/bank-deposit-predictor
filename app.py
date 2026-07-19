import streamlit as st
import numpy as np
import pandas as pd
import json
import xgboost as xgb

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")

st.title("🏦 Term Deposit Conversion Predictor")
st.markdown("Input a customer's specific campaign and economic metrics to calculate their real-time conversion probability.")

# 1. Safely load the feature configuration list
try:
    with open("features.json", "r") as f:
        feature_names = json.load(f)
except FileNotFoundError:
    st.error("Missing 'features.json' file! Please place it in the same directory.")
    st.stop()

# 2. Create the Layout Form for Feature Inputs
st.header("👤 Customer Profile & Campaign Features")

col1, col2 = st.columns(2)

with col1:
    age_group = st.slider("Age Group Code (Scaled/Binned Value)", 0.0, 5.0, 2.0, 0.1)
    education = st.slider("Education Level Code", 0.0, 4.0, 2.0, 0.1)
    previous = st.number_input("Previous Contacts in this Campaign", min_value=0, max_value=10, value=0)
    pdays_cleaned = st.number_input("Days since last contact (Cleaned)", min_value=0, max_value=999, value=999)

with col2:
    # Quick fix: st.sidebar changed to st normal column slider to keep layout neat
    euribor3m = st.slider("Euribor 3-Month Interest Rate", 0.5, 6.0, 3.2, 0.1)
    nr_employed = st.number_input("Number of Employees Indicator", min_value=4000.0, max_value=5300.0, value=5000.0)
    emp_var_rate = st.slider("Employment Variation Rate", -4.0, 2.0, 0.0, 0.1)
    cons_price_idx = st.slider("Consumer Price Index", 90.0, 100.0, 93.5, 0.1)

# 3. Handle Categorical One-Hot Encodings Natively via UI checkboxes
st.subheader("📋 Campaign Context Flags")
c1, c2, c3 = st.columns(3)
with c1:
    poutcome_success = st.checkbox("Previous campaign outcome was a Success")
    marital_married = st.checkbox("Customer is Married")
with c2:
    previously_contacted = st.checkbox("Customer was previously contacted before")
    job_group_white_collar = st.checkbox("Job: White Collar Professional")
with c3:
    contact_telephone = st.checkbox("Contact channel used: Landline Telephone")
    default_unknown = st.checkbox("Credit default status is Unknown")

# 4. Construct the exact input row matching the training shape
input_data = {feat: 0.0 for feat in feature_names}

# Map continuous sliders directly
input_data['age_group'] = age_group
input_data['education'] = education
input_data['previous'] = previous
input_data['pdays_cleaned'] = pdays_cleaned
input_data['euribor3m'] = euribor3m
input_data['nr_employed'] = nr_employed
input_data['emp.var.rate'] = emp_var_rate
input_data['cons.price.idx'] = cons_price_idx

# Map binary fields based on user interactions
input_data['poutcome_success'] = 1.0 if poutcome_success else 0.0
input_data['marital_married'] = 1.0 if marital_married else 0.0
input_data['previously_contacted'] = 1.0 if previously_contacted else 0.0
input_data['job_group_white_collar'] = 1.0 if job_group_white_collar else 0.0
input_data['contact_telephone'] = 1.0 if contact_telephone else 0.0
input_data['default_unknown'] = 1.0 if default_unknown else 0.0

# Convert input dictionary into a proper Pandas DataFrame row structure
df_input = pd.DataFrame([input_data])

# 5. Load the underlying XGBoost model object safely
@st.cache_resource
def load_xgb_model():
    model = xgb.Booster()
    model.load_model("xgb_bank_model.json")
    return model

xgb_engine = load_xgb_model()

# 6. Execute live inference predictions
if st.button("Calculate Subscription Probability", type="primary"):
    dmatrix_input = xgb.DMatrix(df_input)
    raw_probability = float(xgb_engine.predict(dmatrix_input)[0])
    
    st.metric(label="Calculated Probability of Term Deposit Signup", value=f"{raw_probability * 100:.2f}%")
    
    # Establish dynamic baseline classification flag vs the target recall cutoff (0.40)
    if raw_probability >= 0.40:
        st.success("🎯 **Target Match:** High probability prospect. Recommended for active sales queue.")
    else:
        st.warning("💤 **Low Probability Prospect:** Fallback to low-cost automated email campaign.")
