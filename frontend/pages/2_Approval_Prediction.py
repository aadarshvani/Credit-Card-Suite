import streamlit as st
import requests

st.set_page_config(page_title="Approval Prediction", page_icon="💳", layout="wide")

st.title("💳 Credit Card Approval Prediction")
st.markdown("Use the form below to predict if a credit card application should be approved or denied.")

# Group inputs using columns for better layout
with st.form("approval_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)
        amt_income_total = st.number_input("Total Income", min_value=0.0, value=50000.0, step=1000.0)
        days_birth = st.number_input("Days since Birth (negative)", min_value=-30000, max_value=-5000, value=-15000, step=1)
        days_employed = st.number_input("Days Employed (negative)", min_value=-5000, max_value=0, value=-1000, step=1)
        flag_mobil = st.selectbox("Has Mobile Phone", [0, 1])
        flag_work_phone = st.selectbox("Has Work Phone", [0, 1])
        flag_phone = st.selectbox("Has Phone", [0, 1])
        flag_email = st.selectbox("Has Email", [0, 1])

    with col2:
        cnt_fam_members = st.number_input("Number of Family Members", min_value=1, max_value=20, value=2, step=1)
        code_gender = st.selectbox("Gender", ["M", "F"])
        flag_own_car = st.selectbox("Owns Car", [0, 1])
        flag_own_realty = st.selectbox("Owns Realty", [0, 1])
        name_income_type = st.selectbox("Income Type", ["Working", "State servant", "Pensioner", "Student"])
        name_education_type = st.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"])
        name_family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Separated", "Widow"])
        name_housing_type = st.selectbox("Housing Type", ["House / apartment", "Municipal apartment", "Office apartment", "Rented apartment", "With parents"])
        occupation_type = st.selectbox("Occupation Type", ["Working", "State servant", "Pensioner", "Student"])

    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    input_data = {
        "cnt_children": cnt_children,
        "amt_income_total": amt_income_total,
        "days_birth": days_birth,
        "days_employed": days_employed,
        "flag_mobil": flag_mobil,
        "flag_work_phone": flag_work_phone,
        "flag_phone": flag_phone,
        "flag_email": flag_email,
        "cnt_fam_members": cnt_fam_members,
        "code_gender": code_gender,
        "flag_own_car": flag_own_car,
        "flag_own_realty": flag_own_realty,
        "name_income_type": name_income_type,
        "name_education_type": name_education_type,
        "name_family_status": name_family_status,
        "name_housing_type": name_housing_type,
        "occupation_type": occupation_type
    }

    with st.spinner("Making prediction..."):
        try:
            response = requests.post("https://creditcardsuite-backend-latest.onrender.com/api/v1/approval", json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]
                risk_score = result["risk_score"]
                recommendation = result["recommendation"]
                
                if prediction == 0:
                    st.success(f"✅ APPROVE - Low risk application! (Risk Score: {risk_score:.1f}%)")
                else:
                    st.error(f"❌ DENY - High risk application! (Risk Score: {risk_score:.1f}%)")
                
                st.info(f"Probability: {probability:.2f}")
                st.info(f"Recommendation: {recommendation}")
            else:
                st.warning(f"⚠️ Request failed with status code {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}") 