import streamlit as st
import numpy as np
import pandas as pd
import json
import xgboost as xgb

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")

st.title("🏦 Term Deposit Conversion Predictor")
st.markdown("Enter the customer's raw profile details to calculate their real-time subscription probability.")

# 1. Safely load the feature configuration list to guarantee structural alignment
try:
    with open("features.json", "r") as f:
        feature_names = json.load(f)
except FileNotFoundError:
    st.error("Missing 'features.json' file! Please ensure it is generated in your workspace.")
    st.stop()

# 2. Main Raw Intake Form Layout
st.header("👤 Customer Demographics & Profile")
col1, col2 = st.columns(2)

with col1:
    raw_age = st.number_input("Age (Years)", min_value=17, max_value=120, value=35, step=1)
    
    # We display clean titles, but we save lowercased versions internally
    raw_job = st.selectbox("Occupation (Job Type)", options=[
        'Admin', 'Blue-collar', 'Technician', 'Services', 'Management', 
        'Retired', 'Entrepreneur', 'Self-employed', 'Housemaid', 'Unemployed', 'Student', 'unknown'
    ]).lower()
    if raw_job == 'admin': raw_job = 'admin.' # Append the trailing dot required by the raw dataset
    
    raw_marital = st.selectbox("Marital Status", options=['Married', 'Single', 'Divorced', 'unknown']).lower()

with col2:
    # We use mapping to capture custom raw names exactly as required by edu_group_map
    edu_display = st.selectbox("Education Level", options=[
        'University Degree', 'High School', 'Professional Course', 
        'Basic 9y', 'Basic 4y', 'Basic 6y', 'unknown', 'illiterate'
    ])
    edu_ui_map = {
        'University Degree': 'university.degree', 'High School': 'high.school',
        'Professional Course': 'professional.course', 'Basic 9y': 'basic.9y',
        'Basic 4y': 'basic.4y', 'Basic 6y': 'basic.6y', 'unknown': 'unknown', 'illiterate': 'illiterate'
    }
    raw_education = edu_ui_map[edu_display]
    
    raw_default = st.selectbox("Has Credit in Default?", options=['No', 'Yes', 'unknown']).lower()

st.header("📞 Campaign Engagement Details")
col3, col4 = st.columns(2)

with col3:
    raw_campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, max_value=100, value=1, step=1)
    raw_previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=10, value=0, step=1)

with col4:
    raw_pdays = st.number_input("Days Since Last Contact (-1 or 999 if Never Contacted)", min_value=-1, max_value=999, value=999, step=1)
    if raw_pdays == -1:
        raw_pdays = 999

    raw_contact = st.selectbox("Contact Communication Channel", options=['Cellular', 'Telephone']).lower()

st.header("📅 Timing Variables")
col5, col6 = st.columns(2)

with col5:
    raw_month = st.selectbox("Last Contact Month", options=['May', 'Jul', 'Aug', 'Jun', 'Nov', 'Apr', 'Oct', 'Sep', 'Mar', 'Dec']).lower()

with col6:
    raw_day_of_week = st.selectbox("Last Contact Day of the Week", options=['Mon', 'Tue', 'Wed', 'Thu', 'Fri']).lower()

# 3. Macroeconomic Settings (Hidden inside an expander with preset standard dataset averages)
with st.expander("📊 Advanced Macroeconomic Indicators (Auto-Calculated Defaults)"):
    st.caption("These context indicators default to historical averages from the dataset so users don't have to provide them manually.")
    emp_var_rate = st.number_input("Employment Variation Rate (Quarterly)", value=0.08)
    cons_price_idx = st.number_input("Consumer Price Index (Monthly)", value=93.57)
    cons_conf_idx = st.number_input("Consumer Confidence Index (Monthly)", value=-40.50)
    euribor3m = st.number_input("Euribor 3-Month Interest Rate (Daily)", value=3.62)
    nr_employed = st.number_input("Number of Employees Context Indicator", value=5167.0)

# ==================== INTERNAL DATA PROCESSING & TRANSFORMATIONS ====================

# Age Bins -> Mapped
if raw_age <= 20: age_group_str = 'teenager'
elif raw_age <= 35: age_group_str = 'young_adult'
elif raw_age <= 50: age_group_str = 'adult'
else: age_group_str = 'senior'
age_group_val = {'teenager': 0, 'young_adult': 1, 'adult': 2, 'senior': 3}[age_group_str]

# Campaign Bins -> Mapped
if raw_campaign <= 1: campaign_tier_str = 'single_touch'
elif raw_campaign <= 3: campaign_tier_str = 'standard_followup'
elif raw_campaign <= 10: campaign_tier_str = 'high_intensity'
else: campaign_tier_str = 'extreme_outlier'
campaign_tier_val = {'single_touch': 0, 'standard_followup': 1, 'high_intensity': 2, 'extreme_outlier': 3}[campaign_tier_str]

# Education Maps (Fully converted values matching lowercase raw syntax strings)
edu_group_map = {
    'basic.4y': 'primary', 'basic.6y': 'primary', 'basic.9y': 'primary',
    'high.school': 'secondary', 'professional.course': 'tertiary',
    'university.degree': 'graduation', 'illiterate': 'illiterate', 'unknown': 'unknown'
}
edu_ordinal_map = {'illiterate': 0, 'primary': 1, 'secondary': 2, 'unknown': 2, 'graduation': 3, 'tertiary': 4}
education_val = edu_ordinal_map[edu_group_map[raw_education]]

# Job Mapping
job_group_map = {
    'admin.': 'white_collar', 'management': 'white_collar', 'entrepreneur': 'white_collar', 'self-employed': 'white_collar',
    'technician': 'blue_collar', 'blue-collar': 'blue_collar', 'services': 'blue_collar', 'housemaid': 'blue_collar',
    'retired': 'not_working', 'student': 'not_working', 'unemployed': 'not_working', 'unknown': 'not_working'
}
job_group_val = job_group_map[raw_job]

# Month Tier Mapping
month_tier_map = {
    'may': 'high_vol_month', 'jul': 'high_vol_month', 'aug': 'mid_vol_month',
    'jun': 'mid_vol_month', 'nov': 'mid_vol_month', 'apr': 'mid_vol_month',
    'oct': 'low_vol_month', 'sep': 'low_vol_month', 'mar': 'low_vol_month', 'dec': 'low_vol_month'
}
month_tier_val = month_tier_map[raw_month]

# Pdays cleaning rule
previously_contacted = 1 if raw_pdays != 999 else 0
pdays_cleaned = 0 if raw_pdays == 999 else raw_pdays

# B. Internal Feature Dictionary Setup
processed_features = {
    'previous': float(raw_previous),
    'previously_contacted': float(previously_contacted),
    'pdays_cleaned': float(pdays_cleaned),
    'age_group': float(age_group_val),
    'campaign_tier': float(campaign_tier_val),
    'education': float(education_val),
    'emp.var.rate': float(emp_var_rate),
    'cons.price.idx': float(cons_price_idx),
    'cons.conf.idx': float(cons_conf_idx),
    'euribor3m': float(euribor3m),
    'nr.employed': float(nr_employed)
}

# C. Apply Hardcoded StandardScaler values
scaler_stats = {
    'emp.var.rate': {'mean': 0.081886, 'std': 1.570960},
    'cons.price.idx': {'mean': 93.575664, 'std': 0.578840},
    'cons.conf.idx': {'mean': -40.502600, 'std': 4.628198},
    'euribor3m': {'mean': 3.621291, 'std': 1.734447},
    'nr.employed': {'mean': 5167.035911, 'std': 72.251528},
    'pdays_cleaned': {'mean': 0.221200, 'std': 1.378900}
}

for col in scaler_stats:
    processed_features[col] = (processed_features[col] - scaler_stats[col]['mean']) / scaler_stats[col]['std']

# D. Reconstruct Nominals (Forced perfectly lowercase to align with features.json dummy targets)
nominal_selections = {
    f"marital_{raw_marital}": 1,
    f"default_{raw_default}": 1,
    f"contact_{raw_contact}": 1,
    f"day_of_week_{raw_day_of_week}": 1,
    f"month_tier_{month_tier_val}": 1,
    f"job_group_{job_group_val}": 1
}

# Combine and construct input row matching alignment array order
final_input_row = {}
for name in feature_names:
    if name in processed_features:
        final_input_row[name] = processed_features[name]
    elif name in nominal_selections:
        final_input_row[name] = float(nominal_selections[name])
    else:
        final_input_row[name] = 0.0

df_input = pd.DataFrame([final_input_row])

# 4. Model Engine Prediction Execute
@st.cache_resource
def load_xgb_model():
    model = xgb.Booster()
    model.load_model("xgb_bank_model.json")
    return model

xgb_engine = load_xgb_model()

if st.button("Calculate Subscription Probability", type="primary"):
    df_input = df_input[feature_names]
    dmatrix_input = xgb.DMatrix(df_input)
    raw_probability = float(xgb_engine.predict(dmatrix_input)[0])

    st.subheader("Prediction Result")
    st.metric(label="Calculated Probability of Term Deposit Signup", value=f"{raw_probability * 100:.2f}%")

    if raw_probability >= 0.55:
        st.success("🎯 **High Priority Prospect:** High conversion profile. Route immediately to active outbound call agents.")
    else:
        st.warning("💤 **Low Priority Prospect:** Keep in nurture flow via automated low-cost channel campaigns.")
